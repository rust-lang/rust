#rust-clippy
[![Build Status](https://travis-ci.org/Manishearth/rust-clippy.svg?branch=master)](https://travis-ci.org/Manishearth/rust-clippy)

A collection of lints to catch common mistakes and improve your Rust code.

[Jump to usage instructions](#usage)

##Lints
There are 92 lints included in this crate:

name                                                                                                           | default | meaning
---------------------------------------------------------------------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
[approx_constant](https://github.com/Manishearth/rust-clippy/wiki#approx_constant)                             | warn    | the approximate of a known float constant (in `std::f64::consts` or `std::f32::consts`) is found; suggests to use the constant
[bad_bit_mask](https://github.com/Manishearth/rust-clippy/wiki#bad_bit_mask)                                   | warn    | expressions of the form `_ & mask == select` that will only ever return `true` or `false` (because in the example `select` containing bits that `mask` doesn't have)
[block_in_if_condition_expr](https://github.com/Manishearth/rust-clippy/wiki#block_in_if_condition_expr)       | warn    | braces can be eliminated in conditions that are expressions, e.g `if { true } ...`
[block_in_if_condition_stmt](https://github.com/Manishearth/rust-clippy/wiki#block_in_if_condition_stmt)       | warn    | avoid complex blocks in conditions, instead move the block higher and bind it with 'let'; e.g: `if { let x = true; x } ...`
[box_vec](https://github.com/Manishearth/rust-clippy/wiki#box_vec)                                             | warn    | usage of `Box<Vec<T>>`, vector elements are already on the heap
[boxed_local](https://github.com/Manishearth/rust-clippy/wiki#boxed_local)                                     | warn    | using Box<T> where unnecessary
[cast_possible_truncation](https://github.com/Manishearth/rust-clippy/wiki#cast_possible_truncation)           | allow   | casts that may cause truncation of the value, e.g `x as u8` where `x: u32`, or `x as i32` where `x: f32`
[cast_possible_wrap](https://github.com/Manishearth/rust-clippy/wiki#cast_possible_wrap)                       | allow   | casts that may cause wrapping around the value, e.g `x as i32` where `x: u32` and `x > i32::MAX`
[cast_precision_loss](https://github.com/Manishearth/rust-clippy/wiki#cast_precision_loss)                     | allow   | casts that cause loss of precision, e.g `x as f32` where `x: u64`
[cast_sign_loss](https://github.com/Manishearth/rust-clippy/wiki#cast_sign_loss)                               | allow   | casts from signed types to unsigned types, e.g `x as u32` where `x: i32`
[cmp_nan](https://github.com/Manishearth/rust-clippy/wiki#cmp_nan)                                             | deny    | comparisons to NAN (which will always return false, which is probably not intended)
[cmp_owned](https://github.com/Manishearth/rust-clippy/wiki#cmp_owned)                                         | warn    | creating owned instances for comparing with others, e.g. `x == "foo".to_string()`
[collapsible_if](https://github.com/Manishearth/rust-clippy/wiki#collapsible_if)                               | warn    | two nested `if`-expressions can be collapsed into one, e.g. `if x { if y { foo() } }` can be written as `if x && y { foo() }`
[cyclomatic_complexity](https://github.com/Manishearth/rust-clippy/wiki#cyclomatic_complexity)                 | warn    | finds functions that should be split up into multiple functions
[deprecated_semver](https://github.com/Manishearth/rust-clippy/wiki#deprecated_semver)                         | warn    | `Warn` on `#[deprecated(since = "x")]` where x is not semver
[duplicate_underscore_argument](https://github.com/Manishearth/rust-clippy/wiki#duplicate_underscore_argument) | warn    | Function arguments having names which only differ by an underscore
[empty_loop](https://github.com/Manishearth/rust-clippy/wiki#empty_loop)                                       | warn    | empty `loop {}` detected
[eq_op](https://github.com/Manishearth/rust-clippy/wiki#eq_op)                                                 | warn    | equal operands on both sides of a comparison or bitwise combination (e.g. `x == x`)
[explicit_counter_loop](https://github.com/Manishearth/rust-clippy/wiki#explicit_counter_loop)                 | warn    | for-looping with an explicit counter when `_.enumerate()` would do
[explicit_iter_loop](https://github.com/Manishearth/rust-clippy/wiki#explicit_iter_loop)                       | warn    | for-looping over `_.iter()` or `_.iter_mut()` when `&_` or `&mut _` would do
[filter_next](https://github.com/Manishearth/rust-clippy/wiki#filter_next)                                     | warn    | using `filter(p).next()`, which is more succinctly expressed as `.find(p)`
[float_cmp](https://github.com/Manishearth/rust-clippy/wiki#float_cmp)                                         | warn    | using `==` or `!=` on float values (as floating-point operations usually involve rounding errors, it is always better to check for approximate equality within small bounds)
[hashmap_entry](https://github.com/Manishearth/rust-clippy/wiki#hashmap_entry)                                 | warn    | use of `contains_key` followed by `insert` on a `HashMap`
[identity_op](https://github.com/Manishearth/rust-clippy/wiki#identity_op)                                     | warn    | using identity operations, e.g. `x + 0` or `y / 1`
[ineffective_bit_mask](https://github.com/Manishearth/rust-clippy/wiki#ineffective_bit_mask)                   | warn    | expressions where a bit mask will be rendered useless by a comparison, e.g. `(x | 1) > 2`
[inline_always](https://github.com/Manishearth/rust-clippy/wiki#inline_always)                                 | warn    | `#[inline(always)]` is a bad idea in most cases
[iter_next_loop](https://github.com/Manishearth/rust-clippy/wiki#iter_next_loop)                               | warn    | for-looping over `_.next()` which is probably not intended
[len_without_is_empty](https://github.com/Manishearth/rust-clippy/wiki#len_without_is_empty)                   | warn    | traits and impls that have `.len()` but not `.is_empty()`
[len_zero](https://github.com/Manishearth/rust-clippy/wiki#len_zero)                                           | warn    | checking `.len() == 0` or `.len() > 0` (or similar) when `.is_empty()` could be used instead
[let_and_return](https://github.com/Manishearth/rust-clippy/wiki#let_and_return)                               | warn    | creating a let-binding and then immediately returning it like `let x = expr; x` at the end of a block
[let_unit_value](https://github.com/Manishearth/rust-clippy/wiki#let_unit_value)                               | warn    | creating a let binding to a value of unit type, which usually can't be used afterwards
[linkedlist](https://github.com/Manishearth/rust-clippy/wiki#linkedlist)                                       | warn    | usage of LinkedList, usually a vector is faster, or a more specialized data structure like a VecDeque
[map_clone](https://github.com/Manishearth/rust-clippy/wiki#map_clone)                                         | warn    | using `.map(|x| x.clone())` to clone an iterator or option's contents (recommends `.cloned()` instead)
[match_bool](https://github.com/Manishearth/rust-clippy/wiki#match_bool)                                       | warn    | a match on boolean expression; recommends `if..else` block instead
[match_overlapping_arm](https://github.com/Manishearth/rust-clippy/wiki#match_overlapping_arm)                 | warn    | a match has overlapping arms
[match_ref_pats](https://github.com/Manishearth/rust-clippy/wiki#match_ref_pats)                               | warn    | a match or `if let` has all arms prefixed with `&`; the match expression can be dereferenced instead
[min_max](https://github.com/Manishearth/rust-clippy/wiki#min_max)                                             | warn    | `min(_, max(_, _))` (or vice versa) with bounds clamping the result to a constant
[modulo_one](https://github.com/Manishearth/rust-clippy/wiki#modulo_one)                                       | warn    | taking a number modulo 1, which always returns 0
[mut_mut](https://github.com/Manishearth/rust-clippy/wiki#mut_mut)                                             | allow   | usage of double-mut refs, e.g. `&mut &mut ...` (either copy'n'paste error, or shows a fundamental misunderstanding of references)
[mutex_atomic](https://github.com/Manishearth/rust-clippy/wiki#mutex_atomic)                                   | warn    | using a Mutex where an atomic value could be used instead
[mutex_integer](https://github.com/Manishearth/rust-clippy/wiki#mutex_integer)                                 | allow   | using a Mutex for an integer type
[needless_bool](https://github.com/Manishearth/rust-clippy/wiki#needless_bool)                                 | warn    | if-statements with plain booleans in the then- and else-clause, e.g. `if p { true } else { false }`
[needless_lifetimes](https://github.com/Manishearth/rust-clippy/wiki#needless_lifetimes)                       | warn    | using explicit lifetimes for references in function arguments when elision rules would allow omitting them
[needless_range_loop](https://github.com/Manishearth/rust-clippy/wiki#needless_range_loop)                     | warn    | for-looping over a range of indices where an iterator over items would do
[needless_return](https://github.com/Manishearth/rust-clippy/wiki#needless_return)                             | warn    | using a return statement like `return expr;` where an expression would suffice
[needless_update](https://github.com/Manishearth/rust-clippy/wiki#needless_update)                             | warn    | using `{ ..base }` when there are no missing fields
[no_effect](https://github.com/Manishearth/rust-clippy/wiki#no_effect)                                         | warn    | statements with no effect
[non_ascii_literal](https://github.com/Manishearth/rust-clippy/wiki#non_ascii_literal)                         | allow   | using any literal non-ASCII chars in a string literal; suggests using the \\u escape instead
[nonsensical_open_options](https://github.com/Manishearth/rust-clippy/wiki#nonsensical_open_options)           | warn    | nonsensical combination of options for opening a file
[ok_expect](https://github.com/Manishearth/rust-clippy/wiki#ok_expect)                                         | warn    | using `ok().expect()`, which gives worse error messages than calling `expect` directly on the Result
[option_map_unwrap_or](https://github.com/Manishearth/rust-clippy/wiki#option_map_unwrap_or)                   | warn    | using `Option.map(f).unwrap_or(a)`, which is more succinctly expressed as `map_or(a, f)`
[option_map_unwrap_or_else](https://github.com/Manishearth/rust-clippy/wiki#option_map_unwrap_or_else)         | warn    | using `Option.map(f).unwrap_or_else(g)`, which is more succinctly expressed as `map_or_else(g, f)`
[option_unwrap_used](https://github.com/Manishearth/rust-clippy/wiki#option_unwrap_used)                       | allow   | using `Option.unwrap()`, which should at least get a better message using `expect()`
[out_of_bounds_indexing](https://github.com/Manishearth/rust-clippy/wiki#out_of_bounds_indexing)               | deny    | out of bound constant indexing
[panic_params](https://github.com/Manishearth/rust-clippy/wiki#panic_params)                                   | warn    | missing parameters in `panic!`
[precedence](https://github.com/Manishearth/rust-clippy/wiki#precedence)                                       | warn    | catches operations where precedence may be unclear. See the wiki for a list of cases caught
[ptr_arg](https://github.com/Manishearth/rust-clippy/wiki#ptr_arg)                                             | warn    | fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` instead, respectively
[range_step_by_zero](https://github.com/Manishearth/rust-clippy/wiki#range_step_by_zero)                       | warn    | using Range::step_by(0), which produces an infinite iterator
[range_zip_with_len](https://github.com/Manishearth/rust-clippy/wiki#range_zip_with_len)                       | warn    | zipping iterator with a range when enumerate() would do
[redundant_closure](https://github.com/Manishearth/rust-clippy/wiki#redundant_closure)                         | warn    | using redundant closures, i.e. `|a| foo(a)` (which can be written as just `foo`)
[redundant_pattern](https://github.com/Manishearth/rust-clippy/wiki#redundant_pattern)                         | warn    | using `name @ _` in a pattern
[result_unwrap_used](https://github.com/Manishearth/rust-clippy/wiki#result_unwrap_used)                       | allow   | using `Result.unwrap()`, which might be better handled
[reverse_range_loop](https://github.com/Manishearth/rust-clippy/wiki#reverse_range_loop)                       | warn    | Iterating over an empty range, such as `10..0` or `5..5`
[search_is_some](https://github.com/Manishearth/rust-clippy/wiki#search_is_some)                               | warn    | using an iterator search followed by `is_some()`, which is more succinctly expressed as a call to `any()`
[shadow_reuse](https://github.com/Manishearth/rust-clippy/wiki#shadow_reuse)                                   | allow   | rebinding a name to an expression that re-uses the original value, e.g. `let x = x + 1`
[shadow_same](https://github.com/Manishearth/rust-clippy/wiki#shadow_same)                                     | allow   | rebinding a name to itself, e.g. `let mut x = &mut x`
[shadow_unrelated](https://github.com/Manishearth/rust-clippy/wiki#shadow_unrelated)                           | allow   | The name is re-bound without even using the original value
[should_implement_trait](https://github.com/Manishearth/rust-clippy/wiki#should_implement_trait)               | warn    | defining a method that should be implementing a std trait
[single_match](https://github.com/Manishearth/rust-clippy/wiki#single_match)                                   | warn    | a match statement with a single nontrivial arm (i.e, where the other arm is `_ => {}`) is used; recommends `if let` instead
[str_to_string](https://github.com/Manishearth/rust-clippy/wiki#str_to_string)                                 | warn    | using `to_string()` on a str, which should be `to_owned()`
[string_add](https://github.com/Manishearth/rust-clippy/wiki#string_add)                                       | allow   | using `x + ..` where x is a `String`; suggests using `push_str()` instead
[string_add_assign](https://github.com/Manishearth/rust-clippy/wiki#string_add_assign)                         | allow   | using `x = x + ..` where x is a `String`; suggests using `push_str()` instead
[string_to_string](https://github.com/Manishearth/rust-clippy/wiki#string_to_string)                           | warn    | calling `String.to_string()` which is a no-op
[temporary_assignment](https://github.com/Manishearth/rust-clippy/wiki#temporary_assignment)                   | warn    | assignments to temporaries
[toplevel_ref_arg](https://github.com/Manishearth/rust-clippy/wiki#toplevel_ref_arg)                           | warn    | An entire binding was declared as `ref`, in a function argument (`fn foo(ref x: Bar)`), or a `let` statement (`let ref x = foo()`). In such cases, it is preferred to take references with `&`.
[type_complexity](https://github.com/Manishearth/rust-clippy/wiki#type_complexity)                             | warn    | usage of very complex types; recommends factoring out parts into `type` definitions
[unicode_not_nfc](https://github.com/Manishearth/rust-clippy/wiki#unicode_not_nfc)                             | allow   | using a unicode literal not in NFC normal form (see http://www.unicode.org/reports/tr15/ for further information)
[unit_cmp](https://github.com/Manishearth/rust-clippy/wiki#unit_cmp)                                           | warn    | comparing unit values (which is always `true` or `false`, respectively)
[unnecessary_mut_passed](https://github.com/Manishearth/rust-clippy/wiki#unnecessary_mut_passed)               | warn    | an argument is passed as a mutable reference although the function/method only demands an immutable reference
[unneeded_field_pattern](https://github.com/Manishearth/rust-clippy/wiki#unneeded_field_pattern)               | warn    | Struct fields are bound to a wildcard instead of using `..`
[unstable_as_mut_slice](https://github.com/Manishearth/rust-clippy/wiki#unstable_as_mut_slice)                 | warn    | as_mut_slice is not stable and can be replaced by &mut v[..]see https://github.com/rust-lang/rust/issues/27729
[unstable_as_slice](https://github.com/Manishearth/rust-clippy/wiki#unstable_as_slice)                         | warn    | as_slice is not stable and can be replaced by & v[..]see https://github.com/rust-lang/rust/issues/27729
[unused_collect](https://github.com/Manishearth/rust-clippy/wiki#unused_collect)                               | warn    | `collect()`ing an iterator without using the result; this is usually better written as a for loop
[unused_lifetimes](https://github.com/Manishearth/rust-clippy/wiki#unused_lifetimes)                           | warn    | unused lifetimes in function definitions
[used_underscore_binding](https://github.com/Manishearth/rust-clippy/wiki#used_underscore_binding)             | warn    | using a binding which is prefixed with an underscore
[useless_transmute](https://github.com/Manishearth/rust-clippy/wiki#useless_transmute)                         | warn    | transmutes that have the same to and from types
[while_let_loop](https://github.com/Manishearth/rust-clippy/wiki#while_let_loop)                               | warn    | `loop { if let { ... } else break }` can be written as a `while let` loop
[while_let_on_iterator](https://github.com/Manishearth/rust-clippy/wiki#while_let_on_iterator)                 | warn    | using a while-let loop instead of a for loop on an iterator
[wrong_pub_self_convention](https://github.com/Manishearth/rust-clippy/wiki#wrong_pub_self_convention)         | allow   | defining a public method named with an established prefix (like "into_") that takes `self` with the wrong convention
[wrong_self_convention](https://github.com/Manishearth/rust-clippy/wiki#wrong_self_convention)                 | warn    | defining a method named with an established prefix (like "into_") that takes `self` with the wrong convention
[zero_divided_by_zero](https://github.com/Manishearth/rust-clippy/wiki#zero_divided_by_zero)                   | warn    | usage of `0.0 / 0.0` to obtain NaN instead of std::f32::NaN or std::f64::NaN
[zero_width_space](https://github.com/Manishearth/rust-clippy/wiki#zero_width_space)                           | deny    | using a zero-width space in a string literal, which is confusing

More to come, please [file an issue](https://github.com/Manishearth/rust-clippy/issues) if you have ideas!

##Usage

Compiler plugins are highly unstable and will only work with a nightly Rust for now. Since stable Rust is backwards compatible, you should be able to compile your stable programs with nightly Rust with clippy plugged in to circumvent this.

Add in your `Cargo.toml`:
```toml
[dependencies]
clippy = "*"
```

You may also use [`cargo clippy`](https://github.com/arcnmx/cargo-clippy), a custom cargo subcommand that runs clippy on a given project.

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

Produces this warning:
```
src/main.rs:8:5: 11:6 warning: you seem to be trying to use match for destructuring a single type. Consider using `if let`, #[warn(single_match)] on by default
src/main.rs:8     match x {
src/main.rs:9         Some(y) => println!("{:?}", y),
src/main.rs:10         _ => ()
src/main.rs:11     }
src/main.rs:8:5: 11:6 help: Try
if let Some(y) = x { println!("{:?}", y) }
```

You can add options  to `allow`/`warn`/`deny`:
- the whole set of `Warn` lints using the `clippy` lint group (`#![deny(clippy)]`)
- all lints using both the `clippy` and `clippy_pedantic` lint groups (`#![deny(clippy)]`, `#![deny(clippy_pedantic)]`). Note that `clippy_pedantic` contains some very aggressive lints prone to false positives.
- only some lints (`#![deny(single_match, box_vec)]`, etc)
- `allow`/`warn`/`deny` can be limited to a single function or module using `#[allow(...)]`, etc

Note: `deny` produces errors instead of warnings

To have cargo compile your crate with clippy without needing `#![plugin(clippy)]`
in your code, you can use:

```
cargo rustc -- -L /path/to/clippy_so -Z extra-plugins=clippy
```

*[Note](https://github.com/Manishearth/rust-clippy/wiki#a-word-of-warning):* Be sure that clippy was compiled with the same version of rustc that cargo invokes here!

If you want to make clippy an optional dependency, you can do the following:

In your `Cargo.toml`:
```toml
[dependencies]
clippy = {version = "*", optional = true}

[features]
default=[]
```

And, in your `main.rs` or `lib.rs`:

```rust
#![cfg_attr(feature="clippy", feature(plugin))]

#![cfg_attr(feature="clippy", plugin(clippy))]
```

##License
Licensed under [MPL](https://www.mozilla.org/MPL/2.0/). If you're having issues with the license, let me know and I'll try to change it to something more permissive.
