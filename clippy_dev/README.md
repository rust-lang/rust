# Clippy Dev Tool 

The Clippy Dev Tool is a tool to ease Clippy development, similar to `rustc`s `x.py`.

Functionalities (incomplete):

## `lintcheck`
Runs clippy on a fixed set of crates read from `clippy_dev/lintcheck_crates.toml`
and saves logs of the lint warnings into the repo.
We can then check the diff and spot new or disappearing warnings.

From the repo root, run:
````
cargo run --target-dir clippy_dev/target --package clippy_dev \
--bin clippy_dev --manifest-path clippy_dev/Cargo.toml --features lintcheck -- lintcheck
````
or
````
cargo dev-lintcheck
````

By default the logs will be saved into `lintcheck-logs/lintcheck_crates_logs.txt`.

You can set a custom sources.toml by adding `--crates-toml custom.toml`
where `custom.toml` must be a relative path from the repo root.

The results will then be saved to `lintcheck-logs/custom_logs.toml`.

