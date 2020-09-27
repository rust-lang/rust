An `impl Trait` type expands to a recursive type.

Erroneous code example:

```compile_fail,E0720
fn make_recursive_type() -> impl Sized {
    [make_recursive_type(), make_recursive_type()]
}
```

An `impl Trait` type must be expandable to a concrete type that contains no
`impl Trait` types. For example the previous example tries to create an
`impl Trait` type `T` that is equal to `[T, T]`.
