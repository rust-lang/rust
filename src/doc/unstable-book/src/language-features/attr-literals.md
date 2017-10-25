# `attr_literals`

The tracking issue for this feature is: [#34981]

[#34981]: https://github.com/rust-lang/rust/issues/34981

------------------------

At present, literals are only accepted as the value of a key-value pair in
attributes. What's more, only _string_ literals are accepted. This means that
literals can only appear in forms of `#[attr(name = "value")]` or
`#[attr = "value"]`.

The `attr_literals` unstable feature allows other types of literals to be used
in attributes. Here are some examples of attributes that can now be used with
this feature enabled:

```rust,ignore
#[attr]
#[attr(true)]
#[attr(ident)]
#[attr(ident, 100, true, "true", ident = 100, ident = "hello", ident(100))]
#[attr(100)]
#[attr(enabled = true)]
#[enabled(true)]
#[attr("hello")]
#[repr(C, align = 4)]
#[repr(C, align(4))]
```

