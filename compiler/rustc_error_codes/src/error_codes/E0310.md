A parameter type is missing a lifetime constraint or has a lifetime that
does not live long enough.

Erroneous code example:

```compile_fail,E0310
// This won't compile because T is not constrained to the static lifetime
// the reference needs
struct Foo<T> {
    foo: &'static T
}
```

Type parameters in type definitions have lifetimes associated with them that
represent how long the data stored within them is guaranteed to live. This
lifetime must be as long as the data needs to be alive, and missing the
constraint that denotes this will cause this error.

This will compile, because it has the constraint on the type parameter:

```
struct Foo<T: 'static> {
    foo: &'static T
}
```
