An unsafe trait was implemented without an unsafe implementation.

Erroneous code example:

```compile_fail,E0200
struct Foo;

unsafe trait Bar { }

impl Bar for Foo { } // error!
```

Unsafe traits must have unsafe implementations. This error occurs when an
implementation for an unsafe trait isn't marked as unsafe. This may be resolved
by marking the unsafe implementation as unsafe.

```
struct Foo;

unsafe trait Bar { }

unsafe impl Bar for Foo { } // ok!
```
