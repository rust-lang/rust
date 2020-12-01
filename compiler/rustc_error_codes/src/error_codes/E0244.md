#### Note: this error code is no longer emitted by the compiler.

This error indicates that too many type parameters were found in a type or
trait.

For example, the `Foo` struct below has no type parameters, but is supplied
with two in the definition of `Bar`:

```compile_fail,E0107
struct Foo { x: bool }

struct Bar<S, T> { x: Foo<S, T> }
```
