`#[track_caller]` and `#[naked]` cannot both be applied to the same function.

Erroneous code example:

```compile_fail,E0736
#[naked]
#[track_caller]
fn foo() {}
```

This is primarily due to ABI incompatibilities between the two attributes.
See [RFC 2091] for details on this and other limitations.

[RFC 2091]: https://github.com/rust-lang/rfcs/blob/master/text/2091-inline-semantic.md
