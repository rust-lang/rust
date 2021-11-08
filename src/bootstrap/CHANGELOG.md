# Changelog

All notable changes to bootstrap will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [Changes since the last major version]

- `llvm-libunwind` now accepts `in-tree` (formerly true), `system` or `no` (formerly false) [#77703](https://github.com/rust-lang/rust/pull/77703)
- The options `infodir`, `localstatedir`, and `gpg-password-file` are no longer allowed in config.toml. Previously, they were ignored without warning. Note that `infodir` and `localstatedir` are still accepted by `./configure`, with a warning. [#82451](https://github.com/rust-lang/rust/pull/82451)
- Add options for enabling overflow checks, one for std (`overflow-checks-std`) and one for everything else (`overflow-checks`). Both default to false.
- Change the names for `dist` commmands to match the component they generate. [#90684](https://github.com/rust-lang/rust/pull/90684)

### Non-breaking changes

- `x.py check` needs opt-in to check tests (--all-targets) [#77473](https://github.com/rust-lang/rust/pull/77473)
- The default bootstrap profiles are now located at `bootstrap/defaults/config.$PROFILE.toml` (previously they were located at `bootstrap/defaults/config.toml.$PROFILE`) [#77558](https://github.com/rust-lang/rust/pull/77558)
- If you have Rust already installed, `x.py` will now infer the host target
  from the default rust toolchain. [#78513](https://github.com/rust-lang/rust/pull/78513)


## [Version 2] - 2020-09-25

- `host` now defaults to the value of `build` in all cases
  + Previously `host` defaulted to an empty list when `target` was overridden, and to `build` otherwise

### Non-breaking changes

- Add `x.py setup` [#76631](https://github.com/rust-lang/rust/pull/76631)
- Add a changelog for x.py [#76626](https://github.com/rust-lang/rust/pull/76626)
- Optionally, download LLVM from CI on Linux and NixOS. This can be enabled with `download-ci-llvm = true` under `[llvm]`.
  + [#76439](https://github.com/rust-lang/rust/pull/76349)
  + [#76667](https://github.com/rust-lang/rust/pull/76667)
  + [#76708](https://github.com/rust-lang/rust/pull/76708)
- Distribute rustc sources as part of `rustc-dev` [#76856](https://github.com/rust-lang/rust/pull/76856)
- Make the default stage for x.py configurable [#76625](https://github.com/rust-lang/rust/pull/76625). This can be enabled with `build-stage = N`, `doc-stage = N`, etc.
- Add a dedicated debug-logging option [#76588](https://github.com/rust-lang/rust/pull/76588). Previously, `debug-logging` could only be set with `debug-assertions`, slowing down the compiler more than necessary.
- Add sample defaults for x.py [#76628](https://github.com/rust-lang/rust/pull/76628)
- Add `--keep-stage-std`, which behaves like `keep-stage` but allows the stage
  0 compiler artifacts (i.e., stage1/bin/rustc) to be rebuilt if changed
  [#77120](https://github.com/rust-lang/rust/pull/77120).


## [Version 1] - 2020-09-11

This is the first changelog entry, and it does not attempt to be an exhaustive list of features in x.py.
Instead, this documents the changes to bootstrap in the past 2 months.

- Improve defaults in `x.py` [#73964](https://github.com/rust-lang/rust/pull/73964)
  (see [blog post] for details)
- Set `ninja = true` by default [#74922](https://github.com/rust-lang/rust/pull/74922)
- Avoid trying to inversely cross-compile for build triple from host triples [#76415](https://github.com/rust-lang/rust/pull/76415)
- Allow blessing expect-tests in tools [#75975](https://github.com/rust-lang/rust/pull/75975)
- `x.py check` checks tests/examples/benches [#76258](https://github.com/rust-lang/rust/pull/76258)
- Fix `rust.use-lld` when linker is not set [#76326](https://github.com/rust-lang/rust/pull/76326)
- Build tests with LLD if `use-lld = true` was passed [#76378](https://github.com/rust-lang/rust/pull/76378)

[blog post]: https://blog.rust-lang.org/inside-rust/2020/08/30/changes-to-x-py-defaults.html
