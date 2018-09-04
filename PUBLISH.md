Steps to publish a new Clippy version

- Bump `package.version` in `./Cargo.toml` (no need to manually bump `dependencies.clippy_lints.version`).
- Write a changelog entry.
- Run `./pre_publish.sh`
- Review and commit all changed files
- `git push`
- Wait for Travis's approval.
- Merge.
- `cargo publish` in `./clippy_lints`.
- `cargo publish` in the root directory.
- `git pull`.
- `git tag -s v0.0.X -m "v0.0.X"`.
- `git push --tags`.
