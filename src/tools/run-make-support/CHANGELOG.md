# Changelog

All notable changes to the `run_make_support` library should be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the support
library should adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) even if it's
not intended for public consumption (it's moreso to help internally, to help test writers track
changes to the support library).

This support library will probably never reach 1.0. Please bump the minor version in `Cargo.toml` if
you make any breaking changes or other significant changes, or bump the patch version for bug fixes.

## [0.2.0] - 2024-06-11

### Added

- Added `fs_wrapper` module which provides panic-on-fail helpers for their respective `std::fs`
  counterparts, the motivation is to:
    - Reduce littering `.unwrap()` or `.expect()` everywhere for fs operations
    - Help the test writer avoid forgetting to check fs results (even though enforced by
      `-Dunused_must_use`)
    - Provide better panic messages by default
- Added `path()` helper which creates a `Path` relative to `cwd()` (but is less noisy).

### Changed

- Marked many functions with `#[must_use]`, and rmake.rs are now compiled with `-Dunused_must_use`.

## [0.1.0] - 2024-06-09

### Changed

- Use *drop bombs* to enforce that commands are executed; a command invocation will panic if it is
  constructed but never executed. Execution methods `Command::{run, run_fail}` will defuse the drop
  bomb.
- Added `Command` helpers that forward to `std::process::Command` counterparts.

### Removed

- The `env_var` method which was incorrectly named and is `env_clear` underneath and is a footgun
  from `impl_common_helpers`. For example, removing `TMPDIR` on Unix and `TMP`/`TEMP` breaks
  `std::env::temp_dir` and wrecks anything using that, such as rustc's codgen.
- Removed `Deref`/`DerefMut` for `run_make_support::Command` -> `std::process::Command` because it
  causes a method chain like `htmldocck().arg().run()` to fail, because `arg()` resolves to
  `std::process::Command` which also returns a `&mut std::process::Command`, causing the `run()` to
  be not found.

## [0.0.0] - 2024-06-09

Consider this version to contain all changes made to the support library before we started to track
changes in this changelog.

### Added

- Custom command wrappers around `std::process::Command` (`run_make_support::Command`) and custom
  wrapper around `std::process::Output` (`CompletedProcess`) to make it more convenient to work with
  commands and their output, and help avoid forgetting to check for exit status.
    - `Command`: `set_stdin`, `run`, `run_fail`.
    - `CompletedProcess`: `std{err,out}_utf8`, `status`, `assert_std{err,out}_{equals, contains,
      not_contains}`, `assert_exit_code`.
- `impl_common_helpers` macro to avoid repeating adding common convenience methods, including:
    - Environment manipulation methods: `env`, `env_remove`
    - Command argument providers: `arg`, `args`
    - Common invocation inspection (of the command invocation up until `inspect` is called):
      `inspect`
    - Execution methods: `run` (for commands expected to succeed execution, exit status `0`) and
      `run_fail` (for commands expected to fail execution, exit status non-zero).
- Command wrappers around: `rustc`, `clang`, `cc`, `rustc`, `rustdoc`, `llvm-readobj`.
- Thin helpers to construct `python` and `htmldocck` commands.
- `run` and `run_fail` (like `Command::{run, run_fail}`) for running binaries, which sets suitable
  env vars (like `LD_LIB_PATH` or equivalent, `TARGET_RPATH_ENV`, `PATH` on Windows).
- Pseudo command `diff` which has similar functionality as the cli util but not the same API.
- Convenience panic-on-fail helpers `env_var`, `env_var_os`, `cwd` for their `std::env` conterparts.
- Convenience panic-on-fail helpers for reading respective env vars: `target`, `source_root`.
- Platform check helpers: `is_windows`, `is_msvc`, `cygpath_windows`, `uname`.
- fs helpers: `copy_dir_all`.
- `recursive_diff` helper.
- Generic `assert_not_contains` helper.
- Scoped run-with-teardown helper `run_in_tmpdir` which is designed to run commands in a temporary
  directory that is cleared when closure returns.
- Helpers for constructing the name of binaries and libraries: `rust_lib_name`, `static_lib_name`,
  `bin_name`, `dynamic_lib_name`.
- Re-export libraries: `gimli`, `object`, `regex`, `wasmparsmer`.
