Steps to publish a new clippy version

- Bump `package.version` in `./Cargo.toml` (no need to manually bump `dependencies.clippy_lints.version`).
- Write a changelog entry.
- Run `./pre_publish.sh`
- Review and commit all changed files
- `git push`
- Wait for Travis's approval.
- Merge.
- `cargo publish` in `./clippy_clints`.
- `cargo publish` in the root directory.
- `git pull`.
- `git tag -s v0.0.X -m "v0.0.X"`.
- `git push --tags`.
- `git clone git@github.com:rust-lang-nursery/rust-clippy.wiki.git ../rust-clippy.wiki`
- `./util/update_wiki.py`
- `cd ../rust-clippy.wiki`
- `git add *`
- `git commit`
- `git push`
