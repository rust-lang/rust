An inherent implementation was defined for a type outside the current crate.

Erroneous code example:

```compile_fail,E0116
impl Vec<u8> { } // error
```

You can only define an inherent implementation for a type in the same crate
where the type was defined. For example, an `impl` block as above is not allowed
since `Vec` is defined in the standard library.

To fix this problem, you can either:

 - define a trait that has the desired associated functions/types/constants and
   implement the trait for the type in question
 - define a new type wrapping the type and define an implementation on the new
   type

Note that using the `type` keyword does not work here because `type` only
introduces a type alias:

```compile_fail,E0116
type Bytes = Vec<u8>;

impl Bytes { } // error, same as above
```
