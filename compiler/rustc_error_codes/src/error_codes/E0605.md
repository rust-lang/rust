An invalid cast was attempted.

Erroneous code examples:

```compile_fail,E0605
let x = 0u8;
x as Vec<u8>; // error: non-primitive cast: `u8` as `std::vec::Vec<u8>`

// Another example

let v = core::ptr::null::<u8>(); // So here, `v` is a `*const u8`.
v as &u8; // error: non-primitive cast: `*const u8` as `&u8`
```

Only primitive types can be cast into each other. Examples:

```
let x = 0u8;
x as u32; // ok!

let v = core::ptr::null::<u8>();
v as *const i8; // ok!
```

For more information about casts, take a look at the Type cast section in
[The Reference Book][1].

[1]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
