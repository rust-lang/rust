rust-clippy
===========

A collection of lints that give helpful tips to newbies and catch oversights.


Lints included in this crate:

 - `single_match`: Warns when a match statement with a single nontrivial arm (i.e, where the other arm is `_ => {}`) is used, and recommends `if let` instead.
 - `box_vec`: Warns on usage of `Box<Vec<T>>`
 - `dlist`: Warns on usage of `DList`
 - `str_to_string`: Warns on usage of `str::to_string()`
 - `toplevel_ref_arg`: Warns when a function argument is declared `ref` (i.e. `fn foo(ref x: u8)`, but not `fn foo((ref x, ref y): (u8, u8))`)
 - `eq_op`: Warns on equal operands on both sides of a comparison or bitwise combination
 - `bad_bit_mask`: Denies expressions of the form `_ & mask == select` that will only ever return `true` or `false` (because in the example `select` containing bits that `mask` doesn't have)
 - `needless_bool` : Warns on if-statements with plain booleans in the then- and else-clause, e.g. `if p { true } else { false }`
 - `ptr_arg`: Warns on fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` instead, respectively
 - `approx_constant`: Warns if the approximate of a known float constant (in `std::f64::consts` or `std::f32::consts`) is found and suggests to use the constant
 - `cmp_nan`: Denies comparisons to NAN (which will always return false, which is probably not intended)
 - `float_cmp`: Warns on `==` or `!=` comparisons of floaty typed values. As floating-point operations usually involve rounding errors, it is always better to check for approximate equality within some small bounds
 - `precedence`: Warns on expressions where precedence may trip up the unwary reader of the source and suggests adding parenthesis, e.g. `x << 2 + y` will be parsed as `x << (2 + y)`
 - `redundant_closure`: Warns on usage of eta-reducible closures like `|a| foo(a)` (which can be written as just `foo`)

To use, add the following lines to your Cargo.toml:

```
[dependencies]
clippy = "*"
```

In your code, you may add `#![plugin(clippy)]` to use it (you may also need to include a `#![feature(plugin)]` line)

You can allow/warn/deny the whole set using the `clippy` lint group (`#[allow(clippy)]`, etc)

More to come, please [file an issue](https://github.com/Manishearth/rust-clippy/issues) if you have ideas!

Licensed under [MPL](https://www.mozilla.org/MPL/2.0/). If you're having issues with the license, let me know and I'll try to change it to something more permissive.
