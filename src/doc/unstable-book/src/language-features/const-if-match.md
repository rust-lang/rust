# `const_if_match`

The tracking issue for this feature is: [#49146]

[#49146]: https://github.com/rust-lang/rust/issues/49146

------------------------

Allows for the use of conditionals (`if` and `match`) in a const context.
Const contexts include `static`, `static mut`, `const`, `const fn`, const
generics, and array initializers. Enabling this feature flag will also make
`&&` and `||` function normally in a const-context by removing the hack that
replaces them with their non-short-circuiting equivalents, `&` and `|`, in a
`const` or `static`.
