# Testing containers locally

Make sure, you have all required modules unpacked:
```
git submodule update --init
```

Use the `run.sh` from `src/ci/docker`.
The `src/ci/run.sh` is used inside the container.

```
src/ci/docker/run.sh x86_64-gnu-grammartest
```

You can choose one of the targes from `src/ci/docker`.
