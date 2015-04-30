rust-clippy
===========

A collection of lints that give helpful tips to newbies and catch oversights.


Lints included in this crate:

 - `single_match`: Warns when a match statement with a single nontrivial arm (i.e, where the other arm is `_ => {}`) is used, and recommends `if let` instead.
 - `box_vec`: Warns on usage of `Box<Vec<T>>`
 - `dlist`: Warns on usage of `DList`
 - `str_to_string`: Warns on usage of `str::to_string()`
 - `toplevel_ref_arg`: Warns when a function argument is declared `ref` (i.e. `fn foo(ref x: u8)`, but not `fn foo((ref x, ref y): (u8, u8))`).
 - `eq_op`: Warns on equal operands on both sides of a comparison or bitwise combination
 - `bad_bit_mask`: Denies expressions of the form `_ & mask == select` that will only ever return `true` or `false` (because in the example `select` containing bits that `mask` doesn't have)

You can allow/warn/deny the whole set using the `clippy` lint group (`#[allow(clippy)]`, etc)


More to come, please [file an issue](https://github.com/Manishearth/rust-clippy/issues) if you have ideas!

Licensed under [MPL](https://www.mozilla.org/MPL/2.0/). If you're having issues with the license, let me know and I'll try to change it to something more permissive.
