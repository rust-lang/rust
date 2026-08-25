#### Note: this error code is no longer emitted by the compiler.

This error indicates that not enough type parameters were found in a type or
trait.

For example, the `Foo` struct below is defined to be generic in `T`, but the
type parameter is missing in the definition of `Bar`:

```compile_fail,E0107
struct Foo<T> { x: T }

struct Bar { x: Foo }
```
