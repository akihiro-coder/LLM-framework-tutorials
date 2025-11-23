# LLM-framework-tutorials

## Create development's environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip sync uv.lock #環境を統一する
```

## Add new library inside virtual environment

1. requirements.in に追記
2. uv.lock に追記

```bash
uv pip compile requirements.in --generate-hashes -o uv.lock
```

3.ローカル環境を lock と同期

```bash
uv pip sync uv.lock
```
