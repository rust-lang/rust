# Privacy Notes

## LSP server binary

The LSP server performs no network access in itself, but runs `cargo metadata` which will update or download the crate registry and the source code of the project dependencies.

## Visual Studio Code extension

The Code extension connects to GitHub to download updated LSP binaries and, if the nightly channel is selected, to perform update checks.

## Other editor plugins

Any other editor plugins that integrate with `rust-analyzer` are not under the control of the `rust-analyzer` developers. For any privacy concerns, you should check with their respective developers.

## Others

If `cargo check` is enabled (the default), any build scripts or procedural macros used by the project or its dependencies will be executed. This is also the case when `cargo check` is disabled, but build script or procedural macro support is enabled in `rust-analyzer` (off by default).
