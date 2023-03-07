A constant value failed to get evaluated.

Erroneous code example:

```compile_fail,E0080
enum Enum {
    X = (1 << 500),
    Y = (1 / 0),
}
```

This error indicates that the compiler was unable to sensibly evaluate a
constant expression that had to be evaluated. Attempting to divide by 0
or causing an integer overflow are two ways to induce this error.

Ensure that the expressions given can be evaluated as the desired integer type.

See the [Custom Discriminants][custom-discriminants] section of the Reference
for more information about setting custom integer types on fieldless enums
using the [`repr` attribute][repr-attribute].

[custom-discriminants]: https://doc.rust-lang.org/reference/items/enumerations.html#custom-discriminant-values-for-field-less-enumerations
[repr-attribute]: https://doc.rust-lang.org/reference/type-layout.html#reprc-enums
