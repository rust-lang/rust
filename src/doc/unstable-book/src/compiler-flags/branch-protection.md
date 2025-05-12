# `branch-protection`

The tracking issue for this feature is: [#113369](https://github.com/rust-lang/rust/issues/113369).

------------------------

This option lets you enable branch authentication instructions on AArch64.
This option is only accepted when targeting AArch64 architectures.
It takes some combination of the following values, separated by a `,`.

- `pac-ret` - Enable pointer authentication for non-leaf functions.
- `pc` - Use PC as a diversifier using PAuthLR instructions
- `leaf` - Enable pointer authentication for all functions, including leaf functions.
- `b-key` - Sign return addresses with key B, instead of the default key A.
- `bti` - Enable branch target identification.

`leaf`, `b-key` and `pc` are only valid if `pac-ret` was previously specified.
For example, `-Z branch-protection=bti,pac-ret,leaf` is valid, but
`-Z branch-protection=bti,leaf,pac-ret` is not.

Rust's standard library does not ship with BTI or pointer authentication enabled by default.
In Cargo projects the standard library can be recompiled with pointer authentication using the nightly
[build-std](../../cargo/reference/unstable.html#build-std) feature.
