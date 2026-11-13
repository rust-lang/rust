Type parameter defaults cannot use `Self` on structs, enums, or unions.

Erroneous code example:

```compile_fail,E0735
struct Foo<X = Box<Self>> {
    field1: Option<X>,
    field2: Option<X>,
}
// error: type parameters cannot use `Self` in their defaults.
```
