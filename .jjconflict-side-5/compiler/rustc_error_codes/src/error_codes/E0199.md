A trait implementation was marked as unsafe while the trait is safe.

Erroneous code example:

```compile_fail,E0199
struct Foo;

trait Bar { }

unsafe impl Bar for Foo { } // error!
```

Safe traits should not have unsafe implementations, therefore marking an
implementation for a safe trait unsafe will cause a compiler error. Removing
the unsafe marker on the trait noted in the error will resolve this problem:

```
struct Foo;

trait Bar { }

impl Bar for Foo { } // ok!
```
