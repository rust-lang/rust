An auto trait was declared with a method or an associated item.

Erroneous code example:

```compile_fail,E0380
unsafe auto trait Trait {
    type Output; // error!
}
```

Auto traits cannot have methods or associated items. For more information see
the [opt-in builtin traits RFC][RFC 19].

[RFC 19]: https://github.com/rust-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md
