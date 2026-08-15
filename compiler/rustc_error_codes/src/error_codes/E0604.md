A cast to `char` was attempted on a type other than `u8`.

Erroneous code example:

```compile_fail,E0604
0u32 as char; // error: only `u8` can be cast as `char`, not `u32`
```

`char` is a Unicode Scalar Value, an integer value from 0 to 0xD7FF and
0xE000 to 0x10FFFF. (The gap is for surrogate pairs.) Only `u8` always fits in
those ranges so only `u8` may be cast to `char`.

To allow larger values, use `char::from_u32`, which checks the value is valid.

```
assert_eq!(86u8 as char, 'V'); // ok!
assert_eq!(char::from_u32(0x3B1), Some('α')); // ok!
assert_eq!(char::from_u32(0xD800), None); // not a USV.
```

For more information about casts, take a look at the Type cast section in
[The Reference Book][1].

[1]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
