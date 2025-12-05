# Installation

To use rust-analyzer, you need a `rust-analyzer` binary, a text editor
that supports LSP, and the source code of the Rust standard library.

If you're [using VS Code](./vs_code.html), the extension bundles a
copy of the `rust-analyzer` binary. For other editors, you'll need to
[install the binary](./rust_analyzer_binary.html) and [configure your
editor](./other_editors.html).

## Rust Standard Library

rust-analyzer will attempt to install the standard library source code
automatically. You can also install it manually with `rustup`.

    $ rustup component add rust-src

Only the latest stable standard library source is officially supported
for use with rust-analyzer. If you are using an older toolchain or have
an override set, rust-analyzer may fail to understand the Rust source.
You will either need to update your toolchain or use an older version of
rust-analyzer that is compatible with your toolchain.

If you are using an override in your project, you can still force
rust-analyzer to use the stable toolchain via the environment variable
`RUSTUP_TOOLCHAIN`. For example, with VS Code or coc-rust-analyzer:

```json
{ "rust-analyzer.server.extraEnv": { "RUSTUP_TOOLCHAIN": "stable" } }
```

## Crates

There is a package named `ra_ap_rust_analyzer` available on
[crates.io](https://crates.io/crates/ra_ap_rust-analyzer), for people
who want to use rust-analyzer programmatically.

For more details, see [the publish
workflow](https://github.com/rust-lang/rust-analyzer/blob/master/.github/workflows/autopublish.yaml).

