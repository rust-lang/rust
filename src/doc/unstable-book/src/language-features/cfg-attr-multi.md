# `cfg_attr_multi`

The tracking issue for this feature is: [#54881]
The RFC for this feature is: [#2539]

[#54881]: https://github.com/rust-lang/rust/issues/54881
[#2539]: https://github.com/rust-lang/rfcs/pull/2539

------------------------

This feature flag lets you put multiple attributes into a `cfg_attr` attribute.

Example:

```rust,ignore
#[cfg_attr(all(), must_use, optimize)]
```

Because `cfg_attr` resolves before procedural macros, this does not affect
macro resolution at all.