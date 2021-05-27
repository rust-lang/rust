# `native_link_modifiers`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers` feature allows you to use the `modifiers` syntax with the `#[link(..)]` attribute.

Modifiers are specified as a comma-delimited string with each modifier prefixed with either a `+` or `-` to indicate that the modifier is enabled or disabled, respectively. The last boolean value specified for a given modifier wins.
