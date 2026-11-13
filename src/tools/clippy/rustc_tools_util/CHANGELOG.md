# Changelog

## Version 0.4.0

* The commit hashes are now always 10 characters long [#13222](https://github.com/rust-lang/rust-clippy/pull/13222)
* `get_commit_date` and `get_commit_hash` now return `None` if the `git` command fails instead of `Some("")`
  [#13217](https://github.com/rust-lang/rust-clippy/pull/13217)
* `setup_version_info` will now re-run when the git commit changes
  [#13329](https://github.com/rust-lang/rust-clippy/pull/13329)
* New `rerun_if_git_changes` function was added [#13329](https://github.com/rust-lang/rust-clippy/pull/13329)

## Version 0.3.0

* Added `setup_version_info!();` macro for automated scripts.
* `get_version_info!()` no longer requires the user to import `rustc_tools_util::VersionInfo` and `std::env`
