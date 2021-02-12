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

You can set a custom sources.toml by adding `--crates-toml custom.toml` or using `LINTCHECK_TOML="custom.toml"`
where `custom.toml` must be a relative path from the repo root.

The results will then be saved to `lintcheck-logs/custom_logs.toml`.

### configuring the crate sources
The sources to check are saved in a `toml` file.  
There are three types of sources.  
A crates-io source:
````toml
bitflags = {name = "bitflags", versions = ['1.2.1']}
````
Requires a "name" and one or multiple "versions" to be checked.

A git source:
````toml
puffin = {name = "puffin", git_url = "https://github.com/EmbarkStudios/puffin", git_hash = "02dd4a3"}
````
Requires a name, the url to the repo and unique identifier of a commit,
branch or tag which is checked out before linting.  
There is no way to always check `HEAD` because that would lead to changing lint-results as the repo would get updated.  
If `git_url` or `git_hash` is missing, an error will be thrown.

A local dependency:
````toml
 clippy = {name = "clippy", path = "/home/user/clippy"}
````
For when you want to add a repository that is not published yet.  
