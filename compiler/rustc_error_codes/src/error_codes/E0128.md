A type parameter with default value is using forward declared identifier.

Erroneous code example:

```compile_fail,E0128
struct Foo<T = U, U = ()> {
    field1: T,
    field2: U,
}
// error: generic parameters with a default cannot use forward declared
//        identifiers
```

Type parameter defaults can only use parameters that occur before them. Since
type parameters are evaluated in-order, this issue could be fixed by doing:

```
struct Foo<U = (), T = U> {
    field1: T,
    field2: U,
}
```

Please also verify that this wasn't because of a name-clash and rename the type
parameter if so.
