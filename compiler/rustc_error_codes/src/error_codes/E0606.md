An incompatible cast was attempted.

Erroneous code example:

```compile_fail,E0606
let x = &0u8; // Here, `x` is a `&u8`.
let y: u32 = x as u32; // error: casting `&u8` as `u32` is invalid
```

When casting, keep in mind that only primitive types can be cast into each
other. Example:

```
let x = &0u8;
let y: u32 = *x as u32; // We dereference it first and then cast it.
```

For more information about casts, take a look at the Type cast section in
[The Reference Book][1].

[1]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
