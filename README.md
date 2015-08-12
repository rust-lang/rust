#rust-clippy
[![Build Status](https://travis-ci.org/Manishearth/rust-clippy.svg?branch=master)](https://travis-ci.org/Manishearth/rust-clippy)

A collection of lints that give helpful tips to newbies and catch oversights.

##Lints
Lints included in this crate:

 - `single_match`: Warns when a match statement with a single nontrivial arm (i.e, where the other arm is `_ => {}`) is used, and recommends `if let` instead.
 - `box_vec`: Warns on usage of `Box<Vec<T>>`
 - `linkedlist`: Warns on usage of `LinkedList`
 - `str_to_string`: Warns on usage of `str::to_string()`
 - `toplevel_ref_arg`: Warns when a function argument is declared `ref` (i.e. `fn foo(ref x: u8)`, but not `fn foo((ref x, ref y): (u8, u8))`)
 - `eq_op`: Warns on equal operands on both sides of a comparison or bitwise combination
 - `bad_bit_mask`: Denies expressions of the form `_ & mask == select` that will only ever return `true` or `false` (because in the example `select` containing bits that `mask` doesn't have)
 - `ineffective_bit_mask`: Warns on expressions where a bit mask will be rendered useless by a comparison, e.g. `(x | 1) > 2`
 - `needless_bool` : Warns on if-statements with plain booleans in the then- and else-clause, e.g. `if p { true } else { false }`
 - `ptr_arg`: Warns on fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` instead, respectively
 - `approx_constant`: Warns if the approximate of a known float constant (in `std::f64::consts` or `std::f32::consts`) is found and suggests to use the constant
 - `cmp_nan`: Denies comparisons to NAN (which will always return false, which is probably not intended)
 - `float_cmp`: Warns on `==` or `!=` comparisons of floaty typed values. As floating-point operations usually involve rounding errors, it is always better to check for approximate equality within some small bounds
 - `precedence`: Warns on expressions where precedence may trip up the unwary reader of the source and suggests adding parenthesis, e.g. `x << 2 + y` will be parsed as `x << (2 + y)`
 - `redundant_closure`: Warns on usage of eta-reducible closures like `|a| foo(a)` (which can be written as just `foo`)
 - `identity_op`: Warns on identity operations like `x + 0` or `y / 1` (which can be reduced to `x` and `y`, respectively)
 - `mut_mut`: Warns on `&mut &mut` which is either a copy'n'paste error, or shows a fundamental misunderstanding of references
 - `len_zero`: Warns on `_.len() == 0` and suggests using `_.is_empty()` (or similar comparisons with `>` or `!=`)
 - `len_without_is_empty`: Warns on traits or impls that have a `.len()` but no `.is_empty()` method
 - `cmp_owned`: Warns on creating owned instances for comparing with others, e.g. `x == "foo".to_string()`
 - `inline_always`: Warns on `#[inline(always)]`, because in most cases it is a bad idea
 - `collapsible_if`: Warns on cases where two nested `if`-expressions can be collapsed into one, e.g. `if x { if y { foo() } }` can be written as `if x && y { foo() }`
 - `zero_width_space`: Warns on encountering a unicode zero-width space
 - `string_add_assign`: Warns on `x = x + ..` where `x` is a `String` and suggests using `push_str(..)` instead.
 - `string_add`: Matches `x + ..` where `x` is a `String` and where `string_add_assign` doesn't warn. Allowed by default.
 - `needless_return`: Warns on using `return expr;` when a simple `expr` would suffice.
 - `let_and_return`: Warns on doing `let x = expr; x` at the end of a function.
 - `option_unwrap_used`: Warns when `Option.unwrap()` is used, and suggests `.expect()`.
 - `result_unwrap_used`: Warns when `Result.unwrap()` is used (silent by default).
 - `modulo_one`: Warns on taking a number modulo 1, which always has a result of 0.

To use, add the following lines to your Cargo.toml:

```
[dependencies]
clippy = "*"
```

More to come, please [file an issue](https://github.com/Manishearth/rust-clippy/issues) if you have ideas!

##Usage

Compiler plugins are highly unstable and will only work with a nightly Rust for now. Since stable Rust is backwards compatible, you should be able to compile your stable programs with nightly Rust with clippy plugged in to circumvent this.

Add in your `Cargo.toml`:
```toml
[dependencies.clippy]
git = "https://github.com/Manishearth/rust-clippy"
```

Sample `main.rs`:
```rust
#![feature(plugin)]

#![plugin(clippy)]


fn main(){
    let x = Some(1u8);
    match x {
        Some(y) => println!("{:?}", y),
        _ => ()
    }
}
```

Produce this warning:
```
src/main.rs:8:5: 11:6 warning: You seem to be trying to use match for destructuring a single type. Did you mean to use `if let`?, #[warn(single_match)] on by default
src/main.rs:8     match x {
src/main.rs:9         Some(y) => println!("{:?}", y),
src/main.rs:10         _ => ()
src/main.rs:11     }
src/main.rs:8:5: 11:6 help: Try
if let Some(y) = x { println!("{:?}", y) }
```

You can add options  to `allow`/`warn`/`deny`:
- the whole set using the `clippy` lint group (`#![deny(clippy)]`, etc)
- only some lints (`#![deny(single_match, box_vec)]`, etc)
- `allow`/`warn`/`deny` can be limited to a single function or module using `#[allow(...)]`, etc

Note: `deny` produces errors instead of warnings

To have cargo compile your crate with clippy without needing `#![plugin(clippy)]`
in your code, you can use:

```
cargo rustc -- -L /path/to/clippy_so -Z extra-plugins=clippy
```

##License
Licensed under [MPL](https://www.mozilla.org/MPL/2.0/). If you're having issues with the license, let me know and I'll try to change it to something more permissive.
