#### Note: this error code is no longer emitted by the compiler.

A negative impl was added on a trait implementation.

Erroneous code example:

```compile_fail
trait Trait {
    type Bar;
}

struct Foo;

impl !Trait for Foo { } //~ ERROR

fn main() {}
```

Negative impls are only allowed for auto traits. For more
information see the [opt-in builtin traits RFC][RFC 19].

[RFC 19]: https://github.com/rust-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md
