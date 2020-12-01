An inherent implementation was marked unsafe.

Erroneous code example:

```compile_fail,E0197
struct Foo;

unsafe impl Foo { } // error!
```

Inherent implementations (one that do not implement a trait but provide
methods associated with a type) are always safe because they are not
implementing an unsafe trait. Removing the `unsafe` keyword from the inherent
implementation will resolve this error.

```
struct Foo;

impl Foo { } // ok!
```
