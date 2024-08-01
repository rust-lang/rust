## `cargo lintcheck`

Runs Clippy on a fixed set of crates read from
`lintcheck/lintcheck_crates.toml` and saves logs of the lint warnings into the
repo.  We can then check the diff and spot new or disappearing warnings.

From the repo root, run:

```
cargo lintcheck
```

or

```
cargo run --target-dir lintcheck/target --manifest-path lintcheck/Cargo.toml
```

By default, the logs will be saved into
`lintcheck-logs/lintcheck_crates_logs.txt`.

You can set a custom sources.toml by adding `--crates-toml custom.toml` or using
`LINTCHECK_TOML="custom.toml"` where `custom.toml` must be a relative path from
the repo root.

The results will then be saved to `lintcheck-logs/custom_logs.toml`.

The `custom.toml` file may be built using <https://crates.io> recently most
downloaded crates by using `cargo lintcheck popular`. For example, to retrieve
the 200 recently most downloaded crates:

```
cargo lintcheck popular -n 200 custom.toml
```

> Note: Lintcheck isn't sandboxed. Only use it to check crates that you trust or
> sandbox it manually.

### Configuring the Crate Sources

The sources to check are saved in a `toml` file. There are three types of
sources.

1. Crates-io Source

   ```toml
   bitflags = {name = "bitflags", versions = ['1.2.1']}
   ```
   Requires a "name" and one or multiple "versions" to be checked.

2. `git` Source
   ````toml
   puffin = {name = "puffin", git_url = "https://github.com/EmbarkStudios/puffin", git_hash = "02dd4a3"}
   ````
   Requires a name, the url to the repo and unique identifier of a commit,
   branch or tag which is checked out before linting.  There is no way to always
   check `HEAD` because that would lead to changing lint-results as the repo
   would get updated.  If `git_url` or `git_hash` is missing, an error will be
   thrown.

3. Local Dependency
   ```toml
   clippy = {name = "clippy", path = "/home/user/clippy"}
   ```
   For when you want to add a repository that is not published yet.

#### Command Line Options (optional)

```toml
clap = {name = "clap", versions = ['4.5.8'], options = ['-Fderive']}
```

It is possible to specify command line options for each crate. This makes it
possible to enable or disable features.

### Fix mode
You can run `cargo lintcheck --fix` which will run Clippy with `--fix` and
print a warning if Clippy's suggestions fail to apply (if the resulting code does not build). 
This lets us spot bad suggestions or false positives automatically in some cases.  

> Note: Fix mode implies `--all-targets`, so it can fix as much code as it can.

Please note that the target dir should be cleaned afterwards since Clippy will modify
the downloaded sources which can lead to unexpected results when running lintcheck again afterwards.

### Recursive mode
You can run `cargo lintcheck --recursive` to also run Clippy on the dependencies
of the crates listed in the crates source `.toml`. e.g. adding `rand 0.8.5`
would also lint `rand_core`, `rand_chacha`, etc.

Particularly slow crates in the dependency graph can be ignored using
`recursive.ignore`:

```toml
[crates]
cargo = {name = "cargo", versions = ['0.64.0']}

[recursive]
ignore = [
    "unicode-normalization",
]
```
