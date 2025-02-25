# Privacy

The LSP server performs no network access in itself, but runs
`cargo metadata` which will update or download the crate registry and
the source code of the project dependencies. If enabled (the default),
build scripts and procedural macros can do anything.

The Code extension does not access the network.

Any other editor plugins are not under the control of the
`rust-analyzer` developers. For any privacy concerns, you should check
with their respective developers.

For `rust-analyzer` developers, `cargo xtask release` uses the GitHub
API to put together the release notes.
