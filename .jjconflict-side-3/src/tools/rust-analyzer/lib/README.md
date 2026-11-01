# lib

Crates in this directory are published to [crates.io](https://crates.io) and obey semver.

They _could_ live in a separate repo, but we want to experiment with a monorepo setup.

We use these crates from crates.io, not the local copies because we want to ensure that
rust-analyzer works with the versions that are published. This means if you add a new API to these
crates, you need to release a new version to crates.io before you can use that API in rust-analyzer.

To release new versions of these packages, change their version in Cargo.toml. Once your PR is merged into master a workflow will automatically publish the new version to crates.io.

While prototyping, the local versions can be used by uncommenting the relevant lines in the
`[patch.'crates-io']` section in Cargo.toml
