This error indicates that the struct, enum or enum variant cannot be
instantiated from outside of the defining crate as it has been marked
as `non_exhaustive` and as such more fields/variants may be added in
future that could cause adverse side effects for this code.

Erroneous code example:

```ignore (it only works cross-crate)
#[non_exhaustive]
pub struct NormalStruct {
    pub first_field: u16,
    pub second_field: u16,
}

let ns = NormalStruct { first_field: 640, second_field: 480 }; // error!
```

It is recommended that you look for a `new` function or equivalent in the
crate's documentation.
