# `const_in_array_repeat_expressions`

The tracking issue for this feature is: [#49147]

[#49147]: https://github.com/rust-lang/rust/issues/49147

------------------------

Relaxes the rules for repeat expressions, `[x; N]` such that `x` may also be `const` (strictly
speaking rvalue promotable), in addition to `typeof(x): Copy`. The result of `[x; N]` where `x` is
`const` is itself also `const`.
