"""Continue: Songe's models as entries of the `models` list in its
config.yaml (~/.continue, or CONTINUE_GLOBAL_DIR).

Each Songe model is an `openai` provider entry with Songe's endpoint as its
apiBase. The user's own models and their order are kept, so Continue's
default (the model picked in it, else the first one listed) stays theirs.
Continue reads config.json only while there is no config.yaml, so a user
still on config.json is told to move first: a new config.yaml would replace
their whole configuration.
"""

import json
import os
from pathlib import Path

from .. import NAME
from ..models.names import exposed
from .base import Agent, YAMLSectionCodec

CONFIG = "config.yaml"
CODEC = YAMLSectionCodec("models", "name")
# What Continue writes on its first start (core/config/default.ts).
DOCUMENT = {"name": "Main Config", "version": "1.0.0", "schema": "v1"}


def folder(home):
    configured = os.environ.get("CONTINUE_GLOBAL_DIR")
    if configured:
        return Path(configured).absolute()
    return home / ".continue"


def songe_entries(data):
    models = data.get("models")
    if not isinstance(models, dict):
        return {}
    return {k: v for k, v in models.items() if str(k).startswith(f"{NAME}/")}


class Continue(Agent):
    name = "continue"
    pick = "the model selector (songe/…)"
    file_names = (CONFIG,)

    def files(self, home):
        # Found by the folder it makes on its first start, as the others are.
        here = folder(home)
        return [(here / CONFIG, CODEC)] if here.is_dir() else []

    def handles(self, path):
        path = Path(path)
        return path.name == CONFIG and (
            path.parent.name == ".continue"
            or path.parent == folder(Path.home())
        )

    def codec(self, path):
        return CODEC

    def skip_reason(self, home):
        here = folder(home)
        if (here / "config.json").exists() and not (here / CONFIG).exists():
            return (
                "Continue still reads config.json, and a new config.yaml would"
                " replace it · move to config.yaml first"
            )
        return None

    def configure(self, path, data, config, *, default=None, v2=False):
        from .opencode import context_for

        data = {**DOCUMENT, **data}
        models = dict(data.get("models") or {})
        for public in exposed(config):
            context = context_for(config, public)
            entry = dict(models.get(public) or {})
            entry.update(
                name=public,
                provider="openai",
                model=public,
                apiBase=f"http://127.0.0.1:{config.port}/v1",
                apiKey="local",
            )
            # Continue only detects tool calling for model names it knows,
            # and Agent mode needs it.
            entry.setdefault("capabilities", ["tool_use"])
            # maxTokens is left to Continue (35% of the context): its CLI
            # refuses to run when it is half the context or more.
            entry["defaultCompletionOptions"] = {
                **(entry.get("defaultCompletionOptions") or {}),
                "contextLength": context,
            }
            models[public] = entry
        data["models"] = models
        return data

    def owns(self, path, data):
        return bool(songe_entries(data))

    def status(self, path, data, config):
        bases = {entry.get("apiBase") for entry in songe_entries(data).values()}
        if not bases:
            return "missing"
        return "configured" if bases == {f"http://127.0.0.1:{config.port}/v1"} else "differs"

    def on_songe(self, home):
        # The chat model picked in Continue, kept per profile in its global
        # context; with none picked, the first model in config.yaml.
        here = folder(home)
        try:
            context = json.loads((here / "index/globalContext.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            context = {}
        profiles = context.get("selectedModelsByProfileId") if isinstance(context, dict) else None
        picked = [
            chosen.get("chat")
            for chosen in (profiles.values() if isinstance(profiles, dict) else ())
            if isinstance(chosen, dict) and chosen.get("chat")
        ]
        if picked:
            return any(str(name).startswith(f"{NAME}/") for name in picked)
        try:
            models = list(CODEC.load((here / CONFIG).read_bytes()).get("models") or {})
        except (OSError, ValueError):
            return False
        return bool(models) and str(models[0]).startswith(f"{NAME}/")
