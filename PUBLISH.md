Steps to publish a new clippy version

1. `cargo test`.
- Bump `package.version` in `./Cargo.toml` (no need to manually bump `dependencies.clippy_lints.version`).
- Run `./util/update_lints.py`.
- Write a changelog entry.
- Commit `./Cargo.toml`, `./clippy_lints/Cargo.toml` and `./CHANGELOG.md`.
- `git push`
- Wait for Travis's approval.
- Merge.
- `cargo publish` in `./clippy_clints`.
- `cargo publish` in the root directory.
- `git pull`.
- `git tag -s v0.0.X`.
- `git push --tags`.
