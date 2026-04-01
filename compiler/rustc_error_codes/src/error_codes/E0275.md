An evaluation of a trait requirement overflowed.

Erroneous code example:

```compile_fail,E0275
trait Foo {}

struct Bar<T>(T);

impl<T> Foo for T where Bar<T>: Foo {}
```

This error occurs when there was a recursive trait requirement that overflowed
before it could be evaluated. This often means that there is an unbounded
recursion in resolving some type bounds.

To determine if a `T` is `Foo`, we need to check if `Bar<T>` is `Foo`. However,
to do this check, we need to determine that `Bar<Bar<T>>` is `Foo`. To
determine this, we check if `Bar<Bar<Bar<T>>>` is `Foo`, and so on. This is
clearly a recursive requirement that can't be resolved directly.

Consider changing your trait bounds so that they're less self-referential.
