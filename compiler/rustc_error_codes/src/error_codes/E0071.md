A structure-literal syntax was used to create an item that is not a structure
or enum variant.

Example of erroneous code:

```compile_fail,E0071
type U32 = u32;
let t = U32 { value: 4 }; // error: expected struct, variant or union type,
                          // found builtin type `u32`
```

To fix this, ensure that the name was correctly spelled, and that the correct
form of initializer was used.

For example, the code above can be fixed to:

```
type U32 = u32;
let t: U32 = 4;
```

or:

```
struct U32 { value: u32 }
let t = U32 { value: 4 };
```
