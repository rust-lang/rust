`#[track_caller]` requires functions to have the `"Rust"` ABI for implicitly
receiving caller location. See [RFC 2091] for details on this and other
restrictions.

Erroneous code example:

```compile_fail,E0737
#[track_caller]
extern "C" fn foo() {}
```

[RFC 2091]: https://github.com/rust-lang/rfcs/blob/master/text/2091-inline-semantic.md
