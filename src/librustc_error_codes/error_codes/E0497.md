#### Note: this error code is no longer emitted by the compiler.

A stability attribute was used outside of the standard library.

Erroneous code example:

```compile_fail
#[stable] // error: stability attributes may not be used outside of the
          //        standard library
fn foo() {}
```

It is not possible to use stability attributes outside of the standard library.
Also, for now, it is not possible to write deprecation messages either.
