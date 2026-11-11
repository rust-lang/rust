You tried to provide a generic argument to a type which doesn't need it.

Erroneous code example:

```compile_fail,E0109
type X = u32<i32>; // error: type arguments are not allowed for this type
type Y = bool<'static>; // error: lifetime parameters are not allowed on
                        //        this type
```

Check that you used the correct argument and that the definition is correct.

Example:

```
type X = u32; // ok!
type Y = bool; // ok!
```

Note that generic arguments for enum variant constructors go after the variant,
not after the enum. For example, you would write `Option::None::<u32>`,
rather than `Option::<u32>::None`.
