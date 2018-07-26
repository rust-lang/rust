# `doc_alias`

The tracking issue for this feature is: [#50146]

[#50146]: https://github.com/rust-lang/rust/issues/50146

------------------------

You can add alias(es) to an item when using the `rustdoc` search through the
`doc(alias)` attribute. Example:

```rust,no_run
#![feature(doc_alias)]

#[doc(alias = "x")]
#[doc(alias = "big")]
pub struct BigX;
```

Then, when looking for it through the `rustdoc` search, if you enter "x" or
"big", search will show the `BigX` struct first.

Note that this feature is currently hidden behind the `feature(doc_alias)` gate.
