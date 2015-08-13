#rust-clippy
[![Build Status](https://travis-ci.org/Manishearth/rust-clippy.svg?branch=master)](https://travis-ci.org/Manishearth/rust-clippy)

A collection of lints that give helpful tips to newbies and catch oversights.

##Lints
Lints included in this crate:

name                 | default | meaning
---------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
approx_constant      | warn    | the approximate of a known float constant (in `std::f64::consts` or `std::f32::consts`) is found; suggests to use the constant
bad_bit_mask         | deny    | expressions of the form `_ & mask == select` that will only ever return `true` or `false` (because in the example `select` containing bits that `mask` doesn't have)
box_vec              | warn    | usage of `Box<Vec<T>>`, vector elements are already on the heap
cmp_nan              | deny    | comparisons to NAN (which will always return false, which is probably not intended)
cmp_owned            | warn    | creating owned instances for comparing with others, e.g. `x == "foo".to_string()`
collapsible_if       | warn    | two nested `if`-expressions can be collapsed into one, e.g. `if x { if y { foo() } }` can be written as `if x && y { foo() }`
eq_op                | warn    | equal operands on both sides of a comparison or bitwise combination (e.g. `x == x`)
float_cmp            | warn    | using `==` or `!=` on float values (as floating-point operations usually involve rounding errors, it is always better to check for approximate equality within small bounds)
identity_op          | warn    | using identity operations, e.g. `x + 0` or `y / 1`
ineffective_bit_mask | warn    | expressions where a bit mask will be rendered useless by a comparison, e.g. `(x | 1) > 2`
inline_always        | warn    | `#[inline(always)]` is a bad idea in most cases
len_without_is_empty | warn    | traits and impls that have `.len()` but not `.is_empty()`
len_zero             | warn    | checking `.len() == 0` or `.len() > 0` (or similar) when `.is_empty()` could be used instead
let_and_return       | warn    | creating a let-binding and then immediately returning it like `let x = expr; x` at the end of a function
let_unit_value       | warn    | creating a let binding to a value of unit type, which usually can't be used afterwards
linkedlist           | warn    | usage of LinkedList, usually a vector is faster, or a more specialized data structure like a RingBuf
modulo_one           | warn    | taking a number modulo 1, which always returns 0
mut_mut              | warn    | usage of double-mut refs, e.g. `&mut &mut ...` (either copy'n'paste error, or shows a fundamental misunderstanding of references)
needless_bool        | warn    | if-statements with plain booleans in the then- and else-clause, e.g. `if p { true } else { false }`
needless_lifetimes   | warn    | using explicit lifetimes for references in function arguments when elision rules would allow omitting them
needless_range_loop  | warn    | for-looping over a range of indices where an iterator over items would do
needless_return      | warn    | using a return statement like `return expr;` where an expression would suffice
non_ascii_literal    | allow   | using any literal non-ASCII chars in a string literal; suggests using the \\u escape instead
option_unwrap_used   | warn    | using `Option.unwrap()`, which should at least get a better message using `expect()`
precedence           | warn    | expressions where precedence may trip up the unwary reader of the source; suggests adding parentheses, e.g. `x << 2 + y` will be parsed as `x << (2 + y)`
ptr_arg              | allow   | fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` instead, respectively
redundant_closure    | warn    | using redundant closures, i.e. `|a| foo(a)` (which can be written as just `foo`)
result_unwrap_used   | allow   | using `Result.unwrap()`, which might be better handled
single_match         | warn    | a match statement with a single nontrivial arm (i.e, where the other arm is `_ => {}`) is used; recommends `if let` instead
str_to_string        | warn    | using `to_string()` on a str, which should be `to_owned()`
string_add           | allow   | using `x = x + ..` where x is a `String`; suggests using `push_str()` instead
string_add_assign    | allow   | expressions of the form `x = x + ..` where x is a `String`
string_to_string     | warn    | calling `String.to_string()` which is a no-op
toplevel_ref_arg     | warn    | a function argument is declared `ref` (i.e. `fn foo(ref x: u8)`, but not `fn foo((ref x, ref y): (u8, u8))`)
zero_width_space     | deny    | using a zero-width space in a string literal, which is confusing

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
