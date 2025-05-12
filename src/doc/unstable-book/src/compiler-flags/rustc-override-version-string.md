# `RUSTC_OVERRIDE_VERSION_STRING`

This feature is perma-unstable and has no tracking issue.

----

The `RUSTC_OVERRIDE_VERSION_STRING` environment variable overrides the version reported by `rustc --version`. For example:

```console
$ rustc --version
rustc 1.87.0-nightly (43f0014ef 2025-03-25)
$ env RUSTC_OVERRIDE_VERSION_STRING=1.81.0-nightly rustc --version
rustc 1.81.0-nightly
```

Note that the version string is completely overwritten; i.e. rustc discards commit hash and commit date information unless it is explicitly included in the environment variable. The string only applies to the "release" part of the version; for example:
```console
$ RUSTC_OVERRIDE_VERSION_STRING="1.81.0-nightly (aaaaaaaaa 2025-03-22)" rustc -vV
rustc 1.81.0-nightly (aaaaaaaaa 2025-03-22)
binary: rustc
commit-hash: 43f0014ef0f242418674f49052ed39b70f73bc1c
commit-date: 2025-03-25
host: x86_64-unknown-linux-gnu
release: 1.81.0-nightly (aaaaaaaaa 2025-03-22)
LLVM version: 20.1.1
```

Note here that `commit-hash` and `commit-date` do not match the values in the string, and `release` includes the fake hash and date.

This variable has no effect on whether or not unstable features are allowed to be used. It only affects the output of `--version`.

## Why does this environment variable exist?

Various library crates have incomplete or incorrect feature detection.
This environment variable allows bisecting crates that do incorrect detection with `version_check::supports_feature`.

This is not intended to be used for any other case (and, except for bisection, is not particularly useful).

See <https://github.com/rust-lang/rust/pull/124339> for further discussion.
