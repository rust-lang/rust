The Unsize trait should not be implemented directly. All implementations of
Unsize are provided automatically by the compiler.

Erroneous code example:

```compile_fail,E0328
#![feature(unsize)]

use std::marker::Unsize;

pub struct MyType;

impl<T> Unsize<T> for MyType {}
```

If you are defining your own smart pointer type and would like to enable
conversion from a sized to an unsized type with the
[DST coercion system][RFC 982], use [`CoerceUnsized`] instead.

```
#![feature(coerce_unsized)]

use std::ops::CoerceUnsized;

pub struct MyType<T: ?Sized> {
    field_with_unsized_type: T,
}

impl<T, U> CoerceUnsized<MyType<U>> for MyType<T>
    where T: CoerceUnsized<U> {}
```

[RFC 982]: https://github.com/rust-lang/rfcs/blob/master/text/0982-dst-coercion.md
[`CoerceUnsized`]: https://doc.rust-lang.org/std/ops/trait.CoerceUnsized.html
