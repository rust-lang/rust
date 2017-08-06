# rust-clippy

[![Build Status](https://travis-ci.org/rust-lang-nursery/rust-clippy.svg?branch=master)](https://travis-ci.org/rust-lang-nursery/rust-clippy)
[![Windows build status](https://ci.appveyor.com/api/projects/status/github/rust-lang-nursery/rust-clippy?svg=true)](https://ci.appveyor.com/project/rust-lang-nursery/rust-clippy)
[![Clippy Linting Result](http://clippy.bashy.io/github/rust-lang-nursery/rust-clippy/master/badge.svg)](http://clippy.bashy.io/github/rust-lang-nursery/rust-clippy/master/log)
[![Current Version](http://meritbadge.herokuapp.com/clippy)](https://crates.io/crates/clippy)
[![License: MPL-2.0](https://img.shields.io/crates/l/clippy.svg)](#License)

A collection of lints to catch common mistakes and improve your Rust code.

Table of contents:

*   [Lint list](#lints)
*   [Usage instructions](#usage)
*   [Configuration](#configuration)
*   [License](#license)

## Usage

Since this is a tool for helping the developer of a library or application
write better code, it is recommended not to include clippy as a hard dependency.
Options include using it as an optional dependency, as a cargo subcommand, or
as an included feature during build. All of these options are detailed below.

As a general rule clippy will only work with the *latest* Rust nightly for now.

### Optional dependency

If you want to make clippy an optional dependency, you can do the following:

In your `Cargo.toml`:

```toml
[dependencies]
clippy = {version = "*", optional = true}

[features]
default = []
```

And, in your `main.rs` or `lib.rs`:

```rust
#![cfg_attr(feature="clippy", feature(plugin))]

#![cfg_attr(feature="clippy", plugin(clippy))]
```

Then build by enabling the feature: `cargo build --features "clippy"`

Instead of adding the `cfg_attr` attributes you can also run clippy on demand:
`cargo rustc --features clippy -- -Z no-trans -Z extra-plugins=clippy`
(the `-Z no trans`, while not necessary, will stop the compilation process after
typechecking (and lints) have completed, which can significantly reduce the runtime).

### As a cargo subcommand (`cargo clippy`)

An alternate way to use clippy is by installing clippy through cargo as a cargo
subcommand.

```terminal
cargo install clippy
```

Now you can run clippy by invoking `cargo clippy`, or
`rustup run nightly cargo clippy` directly from a directory that is usually
compiled with stable.

In case you are not using rustup, you need to set the environment flag
`SYSROOT` during installation so clippy knows where to find `librustc` and
similar crates.

```terminal
SYSROOT=/path/to/rustc/sysroot cargo install clippy
```

### Running clippy from the command line without installing

To have cargo compile your crate with clippy without needing `#![plugin(clippy)]`
in your code, you can use:

```terminal
cargo rustc -- -L /path/to/clippy_so/dir/ -Z extra-plugins=clippy
```

*[Note](https://github.com/rust-lang-nursery/rust-clippy/wiki#a-word-of-warning):*
Be sure that clippy was compiled with the same version of rustc that cargo invokes here!

### As a Compiler Plugin

*Note:* This is not a recommended installation method.

Since stable Rust is backwards compatible, you should be able to
compile your stable programs with nightly Rust with clippy plugged in to
circumvent this.

Add in your `Cargo.toml`:

```toml
[dependencies]
clippy = "*"
```

You then need to add `#![feature(plugin)]` and `#![plugin(clippy)]` to the top
of your crate entry point (`main.rs` or `lib.rs`).

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

```terminal
src/main.rs:8:5: 11:6 warning: you seem to be trying to use match for destructuring a single type. Consider using `if let`, #[warn(single_match)] on by default
src/main.rs:8     match x {
src/main.rs:9         Some(y) => println!("{:?}", y),
src/main.rs:10         _ => ()
src/main.rs:11     }
src/main.rs:8:5: 11:6 help: Try
if let Some(y) = x { println!("{:?}", y) }
```

## Configuration

Some lints can be configured in a `clippy.toml` file. It contains basic `variable = value` mapping eg.

```toml
blacklisted-names = ["toto", "tata", "titi"]
cyclomatic-complexity-threshold = 30
```

See the wiki for more information about which lints can be configured and the
meaning of the variables.

You can also specify the path to the configuration file with:

```rust
#![plugin(clippy(conf_file="path/to/clippy's/configuration"))]
```

To deactivate the “for further information visit *wiki-link*” message you can
define the `CLIPPY_DISABLE_WIKI_LINKS` environment variable.

### Allowing/denying lints

You can add options  to `allow`/`warn`/`deny`:

*   the whole set of `Warn` lints using the `clippy` lint group (`#![deny(clippy)]`)

*   all lints using both the `clippy` and `clippy_pedantic` lint groups (`#![deny(clippy)]`,
    `#![deny(clippy_pedantic)]`). Note that `clippy_pedantic` contains some very aggressive
    lints prone to false positives.

*   only some lints (`#![deny(single_match, box_vec)]`, etc)

*   `allow`/`warn`/`deny` can be limited to a single function or module using `#[allow(...)]`, etc

Note: `deny` produces errors instead of warnings.

For convenience, `cargo clippy` automatically defines a `cargo-clippy`
features. This lets you set lints level and compile with or without clippy
transparently:

```rust
#[cfg_attr(feature = "cargo-clippy", allow(needless_lifetimes))]
```

## Lints

There are 204 lints included in this crate:

name                                                                                                                         | default | triggers on
-----------------------------------------------------------------------------------------------------------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------
[absurd_extreme_comparisons](https://github.com/rust-lang-nursery/rust-clippy/wiki#absurd_extreme_comparisons)               | warn    | a comparison with a maximum or minimum value that is always true or false
[almost_swapped](https://github.com/rust-lang-nursery/rust-clippy/wiki#almost_swapped)                                       | warn    | `foo = bar; bar = foo` sequence
[approx_constant](https://github.com/rust-lang-nursery/rust-clippy/wiki#approx_constant)                                     | warn    | the approximate of a known float constant (in `std::fXX::consts`)
[assign_op_pattern](https://github.com/rust-lang-nursery/rust-clippy/wiki#assign_op_pattern)                                 | warn    | assigning the result of an operation on a variable to that same variable
[assign_ops](https://github.com/rust-lang-nursery/rust-clippy/wiki#assign_ops)                                               | allow   | any compound assignment operation
[bad_bit_mask](https://github.com/rust-lang-nursery/rust-clippy/wiki#bad_bit_mask)                                           | warn    | expressions of the form `_ & mask == select` that will only ever return `true` or `false`
[blacklisted_name](https://github.com/rust-lang-nursery/rust-clippy/wiki#blacklisted_name)                                   | warn    | usage of a blacklisted/placeholder name
[block_in_if_condition_expr](https://github.com/rust-lang-nursery/rust-clippy/wiki#block_in_if_condition_expr)               | warn    | braces that can be eliminated in conditions, e.g. `if { true } ...`
[block_in_if_condition_stmt](https://github.com/rust-lang-nursery/rust-clippy/wiki#block_in_if_condition_stmt)               | warn    | complex blocks in conditions, e.g. `if { let x = true; x } ...`
[bool_comparison](https://github.com/rust-lang-nursery/rust-clippy/wiki#bool_comparison)                                     | warn    | comparing a variable to a boolean, e.g. `if x == true`
[borrowed_box](https://github.com/rust-lang-nursery/rust-clippy/wiki#borrowed_box)                                           | warn    | a borrow of a boxed type
[box_vec](https://github.com/rust-lang-nursery/rust-clippy/wiki#box_vec)                                                     | warn    | usage of `Box<Vec<T>>`, vector elements are already on the heap
[boxed_local](https://github.com/rust-lang-nursery/rust-clippy/wiki#boxed_local)                                             | warn    | using `Box<T>` where unnecessary
[builtin_type_shadow](https://github.com/rust-lang-nursery/rust-clippy/wiki#builtin_type_shadow)                             | warn    | shadowing a builtin type
[cast_possible_truncation](https://github.com/rust-lang-nursery/rust-clippy/wiki#cast_possible_truncation)                   | allow   | casts that may cause truncation of the value, e.g. `x as u8` where `x: u32`, or `x as i32` where `x: f32`
[cast_possible_wrap](https://github.com/rust-lang-nursery/rust-clippy/wiki#cast_possible_wrap)                               | allow   | casts that may cause wrapping around the value, e.g. `x as i32` where `x: u32` and `x > i32::MAX`
[cast_precision_loss](https://github.com/rust-lang-nursery/rust-clippy/wiki#cast_precision_loss)                             | allow   | casts that cause loss of precision, e.g. `x as f32` where `x: u64`
[cast_sign_loss](https://github.com/rust-lang-nursery/rust-clippy/wiki#cast_sign_loss)                                       | allow   | casts from signed types to unsigned types, e.g. `x as u32` where `x: i32`
[char_lit_as_u8](https://github.com/rust-lang-nursery/rust-clippy/wiki#char_lit_as_u8)                                       | warn    | casting a character literal to u8
[chars_next_cmp](https://github.com/rust-lang-nursery/rust-clippy/wiki#chars_next_cmp)                                       | warn    | using `.chars().next()` to check if a string starts with a char
[clone_double_ref](https://github.com/rust-lang-nursery/rust-clippy/wiki#clone_double_ref)                                   | warn    | using `clone` on `&&T`
[clone_on_copy](https://github.com/rust-lang-nursery/rust-clippy/wiki#clone_on_copy)                                         | warn    | using `clone` on a `Copy` type
[cmp_nan](https://github.com/rust-lang-nursery/rust-clippy/wiki#cmp_nan)                                                     | deny    | comparisons to NAN, which will always return false, probably not intended
[cmp_null](https://github.com/rust-lang-nursery/rust-clippy/wiki#cmp_null)                                                   | warn    | comparing a pointer to a null pointer, suggesting to use `.is_null()` instead.
[cmp_owned](https://github.com/rust-lang-nursery/rust-clippy/wiki#cmp_owned)                                                 | warn    | creating owned instances for comparing with others, e.g. `x == "foo".to_string()`
[collapsible_if](https://github.com/rust-lang-nursery/rust-clippy/wiki#collapsible_if)                                       | warn    | `if`s that can be collapsed (e.g. `if x { if y { ... } }` and `else { if x { ... } }`)
[crosspointer_transmute](https://github.com/rust-lang-nursery/rust-clippy/wiki#crosspointer_transmute)                       | warn    | transmutes that have to or from types that are a pointer to the other
[cyclomatic_complexity](https://github.com/rust-lang-nursery/rust-clippy/wiki#cyclomatic_complexity)                         | warn    | functions that should be split up into multiple functions
[deprecated_semver](https://github.com/rust-lang-nursery/rust-clippy/wiki#deprecated_semver)                                 | warn    | use of `#[deprecated(since = "x")]` where x is not semver
[deref_addrof](https://github.com/rust-lang-nursery/rust-clippy/wiki#deref_addrof)                                           | warn    | use of `*&` or `*&mut` in an expression
[derive_hash_xor_eq](https://github.com/rust-lang-nursery/rust-clippy/wiki#derive_hash_xor_eq)                               | warn    | deriving `Hash` but implementing `PartialEq` explicitly
[diverging_sub_expression](https://github.com/rust-lang-nursery/rust-clippy/wiki#diverging_sub_expression)                   | warn    | whether an expression contains a diverging sub expression
[doc_markdown](https://github.com/rust-lang-nursery/rust-clippy/wiki#doc_markdown)                                           | warn    | presence of `_`, `::` or camel-case outside backticks in documentation
[double_neg](https://github.com/rust-lang-nursery/rust-clippy/wiki#double_neg)                                               | warn    | `--x`, which is a double negation of `x` and not a pre-decrement as in C/C++
[double_parens](https://github.com/rust-lang-nursery/rust-clippy/wiki#double_parens)                                         | warn    | Warn on unnecessary double parentheses
[drop_copy](https://github.com/rust-lang-nursery/rust-clippy/wiki#drop_copy)                                                 | warn    | calls to `std::mem::drop` with a value that implements Copy
[drop_ref](https://github.com/rust-lang-nursery/rust-clippy/wiki#drop_ref)                                                   | warn    | calls to `std::mem::drop` with a reference instead of an owned value
[duplicate_underscore_argument](https://github.com/rust-lang-nursery/rust-clippy/wiki#duplicate_underscore_argument)         | warn    | function arguments having names which only differ by an underscore
[empty_enum](https://github.com/rust-lang-nursery/rust-clippy/wiki#empty_enum)                                               | allow   | enum with no variants
[empty_loop](https://github.com/rust-lang-nursery/rust-clippy/wiki#empty_loop)                                               | warn    | empty `loop {}`, which should block or sleep
[enum_clike_unportable_variant](https://github.com/rust-lang-nursery/rust-clippy/wiki#enum_clike_unportable_variant)         | warn    | C-like enums that are `repr(isize/usize)` and have values that don't fit into an `i32`
[enum_glob_use](https://github.com/rust-lang-nursery/rust-clippy/wiki#enum_glob_use)                                         | allow   | use items that import all variants of an enum
[enum_variant_names](https://github.com/rust-lang-nursery/rust-clippy/wiki#enum_variant_names)                               | warn    | enums where all variants share a prefix/postfix
[eq_op](https://github.com/rust-lang-nursery/rust-clippy/wiki#eq_op)                                                         | warn    | equal operands on both sides of a comparison or bitwise combination (e.g. `x == x`)
[eval_order_dependence](https://github.com/rust-lang-nursery/rust-clippy/wiki#eval_order_dependence)                         | warn    | whether a variable read occurs before a write depends on sub-expression evaluation order
[expl_impl_clone_on_copy](https://github.com/rust-lang-nursery/rust-clippy/wiki#expl_impl_clone_on_copy)                     | warn    | implementing `Clone` explicitly on `Copy` types
[explicit_counter_loop](https://github.com/rust-lang-nursery/rust-clippy/wiki#explicit_counter_loop)                         | warn    | for-looping with an explicit counter when `_.enumerate()` would do
[explicit_into_iter_loop](https://github.com/rust-lang-nursery/rust-clippy/wiki#explicit_into_iter_loop)                     | warn    | for-looping over `_.into_iter()` when `_` would do
[explicit_iter_loop](https://github.com/rust-lang-nursery/rust-clippy/wiki#explicit_iter_loop)                               | warn    | for-looping over `_.iter()` or `_.iter_mut()` when `&_` or `&mut _` would do
[filter_map](https://github.com/rust-lang-nursery/rust-clippy/wiki#filter_map)                                               | allow   | using combinations of `filter`, `map`, `filter_map` and `flat_map` which can usually be written as a single method call
[filter_next](https://github.com/rust-lang-nursery/rust-clippy/wiki#filter_next)                                             | warn    | using `filter(p).next()`, which is more succinctly expressed as `.find(p)`
[float_arithmetic](https://github.com/rust-lang-nursery/rust-clippy/wiki#float_arithmetic)                                   | allow   | any floating-point arithmetic statement
[float_cmp](https://github.com/rust-lang-nursery/rust-clippy/wiki#float_cmp)                                                 | warn    | using `==` or `!=` on float values instead of comparing difference with an epsilon
[for_kv_map](https://github.com/rust-lang-nursery/rust-clippy/wiki#for_kv_map)                                               | warn    | looping on a map using `iter` when `keys` or `values` would do
[for_loop_over_option](https://github.com/rust-lang-nursery/rust-clippy/wiki#for_loop_over_option)                           | warn    | for-looping over an `Option`, which is more clearly expressed as an `if let`
[for_loop_over_result](https://github.com/rust-lang-nursery/rust-clippy/wiki#for_loop_over_result)                           | warn    | for-looping over a `Result`, which is more clearly expressed as an `if let`
[forget_copy](https://github.com/rust-lang-nursery/rust-clippy/wiki#forget_copy)                                             | warn    | calls to `std::mem::forget` with a value that implements Copy
[forget_ref](https://github.com/rust-lang-nursery/rust-clippy/wiki#forget_ref)                                               | warn    | calls to `std::mem::forget` with a reference instead of an owned value
[get_unwrap](https://github.com/rust-lang-nursery/rust-clippy/wiki#get_unwrap)                                               | warn    | using `.get().unwrap()` or `.get_mut().unwrap()` when using `[]` would work instead
[identity_op](https://github.com/rust-lang-nursery/rust-clippy/wiki#identity_op)                                             | warn    | using identity operations, e.g. `x + 0` or `y / 1`
[if_let_redundant_pattern_matching](https://github.com/rust-lang-nursery/rust-clippy/wiki#if_let_redundant_pattern_matching) | warn    | use the proper utility function avoiding an `if let`
[if_let_some_result](https://github.com/rust-lang-nursery/rust-clippy/wiki#if_let_some_result)                               | warn    | usage of `ok()` in `if let Some(pat)` statements is unnecessary, match on `Ok(pat)` instead
[if_not_else](https://github.com/rust-lang-nursery/rust-clippy/wiki#if_not_else)                                             | allow   | `if` branches that could be swapped so no negation operation is necessary on the condition
[if_same_then_else](https://github.com/rust-lang-nursery/rust-clippy/wiki#if_same_then_else)                                 | warn    | if with the same *then* and *else* blocks
[ifs_same_cond](https://github.com/rust-lang-nursery/rust-clippy/wiki#ifs_same_cond)                                         | warn    | consecutive `ifs` with the same condition
[inconsistent_digit_grouping](https://github.com/rust-lang-nursery/rust-clippy/wiki#inconsistent_digit_grouping)             | warn    | integer literals with digits grouped inconsistently
[indexing_slicing](https://github.com/rust-lang-nursery/rust-clippy/wiki#indexing_slicing)                                   | allow   | indexing/slicing usage
[ineffective_bit_mask](https://github.com/rust-lang-nursery/rust-clippy/wiki#ineffective_bit_mask)                           | warn    | expressions where a bit mask will be rendered useless by a comparison, e.g. `(x | 1) > 2`
[inline_always](https://github.com/rust-lang-nursery/rust-clippy/wiki#inline_always)                                         | warn    | use of `#[inline(always)]`
[integer_arithmetic](https://github.com/rust-lang-nursery/rust-clippy/wiki#integer_arithmetic)                               | allow   | any integer arithmetic statement
[invalid_regex](https://github.com/rust-lang-nursery/rust-clippy/wiki#invalid_regex)                                         | deny    | invalid regular expressions
[invalid_upcast_comparisons](https://github.com/rust-lang-nursery/rust-clippy/wiki#invalid_upcast_comparisons)               | allow   | a comparison involving an upcast which is always true or false
[items_after_statements](https://github.com/rust-lang-nursery/rust-clippy/wiki#items_after_statements)                       | allow   | blocks where an item comes after a statement
[iter_cloned_collect](https://github.com/rust-lang-nursery/rust-clippy/wiki#iter_cloned_collect)                             | warn    | using `.cloned().collect()` on slice to create a `Vec`
[iter_next_loop](https://github.com/rust-lang-nursery/rust-clippy/wiki#iter_next_loop)                                       | warn    | for-looping over `_.next()` which is probably not intended
[iter_nth](https://github.com/rust-lang-nursery/rust-clippy/wiki#iter_nth)                                                   | warn    | using `.iter().nth()` on a standard library type with O(1) element access
[iter_skip_next](https://github.com/rust-lang-nursery/rust-clippy/wiki#iter_skip_next)                                       | warn    | using `.skip(x).next()` on an iterator
[iterator_step_by_zero](https://github.com/rust-lang-nursery/rust-clippy/wiki#iterator_step_by_zero)                         | warn    | using `Iterator::step_by(0)`, which produces an infinite iterator
[large_digit_groups](https://github.com/rust-lang-nursery/rust-clippy/wiki#large_digit_groups)                               | warn    | grouping digits into groups that are too large
[large_enum_variant](https://github.com/rust-lang-nursery/rust-clippy/wiki#large_enum_variant)                               | warn    | large size difference between variants on an enum
[len_without_is_empty](https://github.com/rust-lang-nursery/rust-clippy/wiki#len_without_is_empty)                           | warn    | traits or impls with a public `len` method but no corresponding `is_empty` method
[len_zero](https://github.com/rust-lang-nursery/rust-clippy/wiki#len_zero)                                                   | warn    | checking `.len() == 0` or `.len() > 0` (or similar) when `.is_empty()` could be used instead
[let_and_return](https://github.com/rust-lang-nursery/rust-clippy/wiki#let_and_return)                                       | warn    | creating a let-binding and then immediately returning it like `let x = expr; x` at the end of a block
[let_unit_value](https://github.com/rust-lang-nursery/rust-clippy/wiki#let_unit_value)                                       | warn    | creating a let binding to a value of unit type, which usually can't be used afterwards
[linkedlist](https://github.com/rust-lang-nursery/rust-clippy/wiki#linkedlist)                                               | warn    | usage of LinkedList, usually a vector is faster, or a more specialized data structure like a VecDeque
[logic_bug](https://github.com/rust-lang-nursery/rust-clippy/wiki#logic_bug)                                                 | warn    | boolean expressions that contain terminals which can be eliminated
[manual_swap](https://github.com/rust-lang-nursery/rust-clippy/wiki#manual_swap)                                             | warn    | manual swap of two variables
[many_single_char_names](https://github.com/rust-lang-nursery/rust-clippy/wiki#many_single_char_names)                       | warn    | too many single character bindings
[map_clone](https://github.com/rust-lang-nursery/rust-clippy/wiki#map_clone)                                                 | warn    | using `.map(|x| x.clone())` to clone an iterator or option's contents
[map_entry](https://github.com/rust-lang-nursery/rust-clippy/wiki#map_entry)                                                 | warn    | use of `contains_key` followed by `insert` on a `HashMap` or `BTreeMap`
[match_bool](https://github.com/rust-lang-nursery/rust-clippy/wiki#match_bool)                                               | warn    | a match on a boolean expression instead of an `if..else` block
[match_overlapping_arm](https://github.com/rust-lang-nursery/rust-clippy/wiki#match_overlapping_arm)                         | warn    | a match with overlapping arms
[match_ref_pats](https://github.com/rust-lang-nursery/rust-clippy/wiki#match_ref_pats)                                       | warn    | a match or `if let` with all arms prefixed with `&` instead of deref-ing the match expression
[match_same_arms](https://github.com/rust-lang-nursery/rust-clippy/wiki#match_same_arms)                                     | warn    | `match` with identical arm bodies
[match_wild_err_arm](https://github.com/rust-lang-nursery/rust-clippy/wiki#match_wild_err_arm)                               | warn    | a match with `Err(_)` arm and take drastic actions
[mem_forget](https://github.com/rust-lang-nursery/rust-clippy/wiki#mem_forget)                                               | allow   | `mem::forget` usage on `Drop` types, likely to cause memory leaks
[min_max](https://github.com/rust-lang-nursery/rust-clippy/wiki#min_max)                                                     | warn    | `min(_, max(_, _))` (or vice versa) with bounds clamping the result to a constant
[misrefactored_assign_op](https://github.com/rust-lang-nursery/rust-clippy/wiki#misrefactored_assign_op)                     | warn    | having a variable on both sides of an assign op
[missing_docs_in_private_items](https://github.com/rust-lang-nursery/rust-clippy/wiki#missing_docs_in_private_items)         | allow   | detects missing documentation for public and private members
[mixed_case_hex_literals](https://github.com/rust-lang-nursery/rust-clippy/wiki#mixed_case_hex_literals)                     | warn    | hex literals whose letter digits are not consistently upper- or lowercased
[module_inception](https://github.com/rust-lang-nursery/rust-clippy/wiki#module_inception)                                   | warn    | modules that have the same name as their parent module
[modulo_one](https://github.com/rust-lang-nursery/rust-clippy/wiki#modulo_one)                                               | warn    | taking a number modulo 1, which always returns 0
[mut_from_ref](https://github.com/rust-lang-nursery/rust-clippy/wiki#mut_from_ref)                                           | warn    | fns that create mutable refs from immutable ref args
[mut_mut](https://github.com/rust-lang-nursery/rust-clippy/wiki#mut_mut)                                                     | allow   | usage of double-mut refs, e.g. `&mut &mut ...`
[mutex_atomic](https://github.com/rust-lang-nursery/rust-clippy/wiki#mutex_atomic)                                           | warn    | using a mutex where an atomic value could be used instead
[mutex_integer](https://github.com/rust-lang-nursery/rust-clippy/wiki#mutex_integer)                                         | allow   | using a mutex for an integer type
[needless_bool](https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_bool)                                         | warn    | if-statements with plain booleans in the then- and else-clause, e.g. `if p { true } else { false }`
[needless_borrow](https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_borrow)                                     | warn    | taking a reference that is going to be automatically dereferenced
[needless_borrowed_reference](https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_borrowed_reference)             | warn    | taking a needless borrowed reference
[needless_continue](https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_continue)                                 | warn    | `continue` statements that can be replaced by a rearrangement of code
[needless_lifetimes](https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_lifetimes)                               | warn    | using explicit lifetimes for references in function arguments when elision rules would allow omitting them
[needless_pass_by_value](https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_pass_by_value)                       | warn    | functions taking arguments by value, but not consuming them in its body
[needless_range_loop](https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_range_loop)                             | warn    | for-looping over a range of indices where an iterator over items would do
[needless_return](https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_return)                                     | warn    | using a return statement like `return expr;` where an expression would suffice
[needless_update](https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_update)                                     | warn    | using `Foo { ..base }` when there are no missing fields
[neg_multiply](https://github.com/rust-lang-nursery/rust-clippy/wiki#neg_multiply)                                           | warn    | multiplying integers with -1
[never_loop](https://github.com/rust-lang-nursery/rust-clippy/wiki#never_loop)                                               | warn    | any loop that will always `break` or `return`
[new_ret_no_self](https://github.com/rust-lang-nursery/rust-clippy/wiki#new_ret_no_self)                                     | warn    | not returning `Self` in a `new` method
[new_without_default](https://github.com/rust-lang-nursery/rust-clippy/wiki#new_without_default)                             | warn    | `fn new() -> Self` method without `Default` implementation
[new_without_default_derive](https://github.com/rust-lang-nursery/rust-clippy/wiki#new_without_default_derive)               | warn    | `fn new() -> Self` without `#[derive]`able `Default` implementation
[no_effect](https://github.com/rust-lang-nursery/rust-clippy/wiki#no_effect)                                                 | warn    | statements with no effect
[non_ascii_literal](https://github.com/rust-lang-nursery/rust-clippy/wiki#non_ascii_literal)                                 | allow   | using any literal non-ASCII chars in a string literal instead of using the `\\u` escape
[nonminimal_bool](https://github.com/rust-lang-nursery/rust-clippy/wiki#nonminimal_bool)                                     | allow   | boolean expressions that can be written more concisely
[nonsensical_open_options](https://github.com/rust-lang-nursery/rust-clippy/wiki#nonsensical_open_options)                   | warn    | nonsensical combination of options for opening a file
[not_unsafe_ptr_arg_deref](https://github.com/rust-lang-nursery/rust-clippy/wiki#not_unsafe_ptr_arg_deref)                   | warn    | public functions dereferencing raw pointer arguments but not marked `unsafe`
[ok_expect](https://github.com/rust-lang-nursery/rust-clippy/wiki#ok_expect)                                                 | warn    | using `ok().expect()`, which gives worse error messages than calling `expect` directly on the Result
[op_ref](https://github.com/rust-lang-nursery/rust-clippy/wiki#op_ref)                                                       | warn    | taking a reference to satisfy the type constraints on `==`
[option_map_unwrap_or](https://github.com/rust-lang-nursery/rust-clippy/wiki#option_map_unwrap_or)                           | allow   | using `Option.map(f).unwrap_or(a)`, which is more succinctly expressed as `map_or(a, f)`
[option_map_unwrap_or_else](https://github.com/rust-lang-nursery/rust-clippy/wiki#option_map_unwrap_or_else)                 | allow   | using `Option.map(f).unwrap_or_else(g)`, which is more succinctly expressed as `map_or_else(g, f)`
[option_unwrap_used](https://github.com/rust-lang-nursery/rust-clippy/wiki#option_unwrap_used)                               | allow   | using `Option.unwrap()`, which should at least get a better message using `expect()`
[or_fun_call](https://github.com/rust-lang-nursery/rust-clippy/wiki#or_fun_call)                                             | warn    | using any `*or` method with a function call, which suggests `*or_else`
[out_of_bounds_indexing](https://github.com/rust-lang-nursery/rust-clippy/wiki#out_of_bounds_indexing)                       | deny    | out of bounds constant indexing
[overflow_check_conditional](https://github.com/rust-lang-nursery/rust-clippy/wiki#overflow_check_conditional)               | warn    | overflow checks inspired by C which are likely to panic
[panic_params](https://github.com/rust-lang-nursery/rust-clippy/wiki#panic_params)                                           | warn    | missing parameters in `panic!` calls
[partialeq_ne_impl](https://github.com/rust-lang-nursery/rust-clippy/wiki#partialeq_ne_impl)                                 | warn    | re-implementing `PartialEq::ne`
[possible_missing_comma](https://github.com/rust-lang-nursery/rust-clippy/wiki#possible_missing_comma)                       | warn    | possible missing comma in array
[precedence](https://github.com/rust-lang-nursery/rust-clippy/wiki#precedence)                                               | warn    | operations where precedence may be unclear
[print_stdout](https://github.com/rust-lang-nursery/rust-clippy/wiki#print_stdout)                                           | allow   | printing on stdout
[print_with_newline](https://github.com/rust-lang-nursery/rust-clippy/wiki#print_with_newline)                               | warn    | using `print!()` with a format string that ends in a newline
[ptr_arg](https://github.com/rust-lang-nursery/rust-clippy/wiki#ptr_arg)                                                     | warn    | fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` instead, respectively
[pub_enum_variant_names](https://github.com/rust-lang-nursery/rust-clippy/wiki#pub_enum_variant_names)                       | allow   | enums where all variants share a prefix/postfix
[range_zip_with_len](https://github.com/rust-lang-nursery/rust-clippy/wiki#range_zip_with_len)                               | warn    | zipping iterator with a range when `enumerate()` would do
[redundant_closure](https://github.com/rust-lang-nursery/rust-clippy/wiki#redundant_closure)                                 | warn    | redundant closures, i.e. `|a| foo(a)` (which can be written as just `foo`)
[redundant_closure_call](https://github.com/rust-lang-nursery/rust-clippy/wiki#redundant_closure_call)                       | warn    | throwaway closures called in the expression they are defined
[redundant_pattern](https://github.com/rust-lang-nursery/rust-clippy/wiki#redundant_pattern)                                 | warn    | using `name @ _` in a pattern
[regex_macro](https://github.com/rust-lang-nursery/rust-clippy/wiki#regex_macro)                                             | warn    | use of `regex!(_)` instead of `Regex::new(_)`
[result_unwrap_used](https://github.com/rust-lang-nursery/rust-clippy/wiki#result_unwrap_used)                               | allow   | using `Result.unwrap()`, which might be better handled
[reverse_range_loop](https://github.com/rust-lang-nursery/rust-clippy/wiki#reverse_range_loop)                               | warn    | iteration over an empty range, such as `10..0` or `5..5`
[search_is_some](https://github.com/rust-lang-nursery/rust-clippy/wiki#search_is_some)                                       | warn    | using an iterator search followed by `is_some()`, which is more succinctly expressed as a call to `any()`
[serde_api_misuse](https://github.com/rust-lang-nursery/rust-clippy/wiki#serde_api_misuse)                                   | warn    | various things that will negatively affect your serde experience
[shadow_reuse](https://github.com/rust-lang-nursery/rust-clippy/wiki#shadow_reuse)                                           | allow   | rebinding a name to an expression that re-uses the original value, e.g. `let x = x + 1`
[shadow_same](https://github.com/rust-lang-nursery/rust-clippy/wiki#shadow_same)                                             | allow   | rebinding a name to itself, e.g. `let mut x = &mut x`
[shadow_unrelated](https://github.com/rust-lang-nursery/rust-clippy/wiki#shadow_unrelated)                                   | allow   | rebinding a name without even using the original value
[short_circuit_statement](https://github.com/rust-lang-nursery/rust-clippy/wiki#short_circuit_statement)                     | warn    | using a short circuit boolean condition as a statement
[should_assert_eq](https://github.com/rust-lang-nursery/rust-clippy/wiki#should_assert_eq)                                   | warn    | using `assert` macro for asserting equality
[should_implement_trait](https://github.com/rust-lang-nursery/rust-clippy/wiki#should_implement_trait)                       | warn    | defining a method that should be implementing a std trait
[similar_names](https://github.com/rust-lang-nursery/rust-clippy/wiki#similar_names)                                         | allow   | similarly named items and bindings
[single_char_pattern](https://github.com/rust-lang-nursery/rust-clippy/wiki#single_char_pattern)                             | warn    | using a single-character str where a char could be used, e.g. `_.split("x")`
[single_match](https://github.com/rust-lang-nursery/rust-clippy/wiki#single_match)                                           | warn    | a match statement with a single nontrivial arm (i.e. where the other arm is `_ => {}`) instead of `if let`
[single_match_else](https://github.com/rust-lang-nursery/rust-clippy/wiki#single_match_else)                                 | allow   | a match statement with a two arms where the second arm's pattern is a wildcard instead of `if let`
[string_add](https://github.com/rust-lang-nursery/rust-clippy/wiki#string_add)                                               | allow   | using `x + ..` where x is a `String` instead of `push_str()`
[string_add_assign](https://github.com/rust-lang-nursery/rust-clippy/wiki#string_add_assign)                                 | allow   | using `x = x + ..` where x is a `String` instead of `push_str()`
[string_extend_chars](https://github.com/rust-lang-nursery/rust-clippy/wiki#string_extend_chars)                             | warn    | using `x.extend(s.chars())` where s is a `&str` or `String`
[string_lit_as_bytes](https://github.com/rust-lang-nursery/rust-clippy/wiki#string_lit_as_bytes)                             | warn    | calling `as_bytes` on a string literal instead of using a byte string literal
[stutter](https://github.com/rust-lang-nursery/rust-clippy/wiki#stutter)                                                     | allow   | type names prefixed/postfixed with their containing module's name
[suspicious_assignment_formatting](https://github.com/rust-lang-nursery/rust-clippy/wiki#suspicious_assignment_formatting)   | warn    | suspicious formatting of `*=`, `-=` or `!=`
[suspicious_else_formatting](https://github.com/rust-lang-nursery/rust-clippy/wiki#suspicious_else_formatting)               | warn    | suspicious formatting of `else if`
[temporary_assignment](https://github.com/rust-lang-nursery/rust-clippy/wiki#temporary_assignment)                           | warn    | assignments to temporaries
[temporary_cstring_as_ptr](https://github.com/rust-lang-nursery/rust-clippy/wiki#temporary_cstring_as_ptr)                   | warn    | getting the inner pointer of a temporary `CString`
[too_many_arguments](https://github.com/rust-lang-nursery/rust-clippy/wiki#too_many_arguments)                               | warn    | functions with too many arguments
[toplevel_ref_arg](https://github.com/rust-lang-nursery/rust-clippy/wiki#toplevel_ref_arg)                                   | warn    | an entire binding declared as `ref`, in a function argument or a `let` statement
[transmute_ptr_to_ref](https://github.com/rust-lang-nursery/rust-clippy/wiki#transmute_ptr_to_ref)                           | warn    | transmutes from a pointer to a reference type
[trivial_regex](https://github.com/rust-lang-nursery/rust-clippy/wiki#trivial_regex)                                         | warn    | trivial regular expressions
[type_complexity](https://github.com/rust-lang-nursery/rust-clippy/wiki#type_complexity)                                     | warn    | usage of very complex types that might be better factored into `type` definitions
[unicode_not_nfc](https://github.com/rust-lang-nursery/rust-clippy/wiki#unicode_not_nfc)                                     | allow   | using a unicode literal not in NFC normal form (see [unicode tr15](http://www.unicode.org/reports/tr15/) for further information)
[unit_cmp](https://github.com/rust-lang-nursery/rust-clippy/wiki#unit_cmp)                                                   | warn    | comparing unit values
[unnecessary_cast](https://github.com/rust-lang-nursery/rust-clippy/wiki#unnecessary_cast)                                   | warn    | cast to the same type, e.g. `x as i32` where `x: i32`
[unnecessary_mut_passed](https://github.com/rust-lang-nursery/rust-clippy/wiki#unnecessary_mut_passed)                       | warn    | an argument passed as a mutable reference although the callee only demands an immutable reference
[unnecessary_operation](https://github.com/rust-lang-nursery/rust-clippy/wiki#unnecessary_operation)                         | warn    | outer expressions with no effect
[unneeded_field_pattern](https://github.com/rust-lang-nursery/rust-clippy/wiki#unneeded_field_pattern)                       | warn    | struct fields bound to a wildcard instead of using `..`
[unreadable_literal](https://github.com/rust-lang-nursery/rust-clippy/wiki#unreadable_literal)                               | warn    | long integer literal without underscores
[unsafe_removed_from_name](https://github.com/rust-lang-nursery/rust-clippy/wiki#unsafe_removed_from_name)                   | warn    | `unsafe` removed from API names on import
[unseparated_literal_suffix](https://github.com/rust-lang-nursery/rust-clippy/wiki#unseparated_literal_suffix)               | allow   | literals whose suffix is not separated by an underscore
[unused_collect](https://github.com/rust-lang-nursery/rust-clippy/wiki#unused_collect)                                       | warn    | `collect()`ing an iterator without using the result; this is usually better written as a for loop
[unused_io_amount](https://github.com/rust-lang-nursery/rust-clippy/wiki#unused_io_amount)                                   | deny    | unused written/read amount
[unused_label](https://github.com/rust-lang-nursery/rust-clippy/wiki#unused_label)                                           | warn    | unused labels
[unused_lifetimes](https://github.com/rust-lang-nursery/rust-clippy/wiki#unused_lifetimes)                                   | warn    | unused lifetimes in function definitions
[use_debug](https://github.com/rust-lang-nursery/rust-clippy/wiki#use_debug)                                                 | allow   | use of `Debug`-based formatting
[used_underscore_binding](https://github.com/rust-lang-nursery/rust-clippy/wiki#used_underscore_binding)                     | allow   | using a binding which is prefixed with an underscore
[useless_attribute](https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_attribute)                                 | warn    | use of lint attributes on `extern crate` items
[useless_format](https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_format)                                       | warn    | useless use of `format!`
[useless_let_if_seq](https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_let_if_seq)                               | warn    | unidiomatic `let mut` declaration followed by initialization in `if`
[useless_transmute](https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_transmute)                                 | warn    | transmutes that have the same to and from types or could be a cast/coercion
[useless_vec](https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_vec)                                             | warn    | useless `vec!`
[verbose_bit_mask](https://github.com/rust-lang-nursery/rust-clippy/wiki#verbose_bit_mask)                                   | warn    | expressions where a bit mask is less readable than the corresponding method call
[while_let_loop](https://github.com/rust-lang-nursery/rust-clippy/wiki#while_let_loop)                                       | warn    | `loop { if let { ... } else break }`, which can be written as a `while let` loop
[while_let_on_iterator](https://github.com/rust-lang-nursery/rust-clippy/wiki#while_let_on_iterator)                         | warn    | using a while-let loop instead of a for loop on an iterator
[wrong_pub_self_convention](https://github.com/rust-lang-nursery/rust-clippy/wiki#wrong_pub_self_convention)                 | allow   | defining a public method named with an established prefix (like "into_") that takes `self` with the wrong convention
[wrong_self_convention](https://github.com/rust-lang-nursery/rust-clippy/wiki#wrong_self_convention)                         | warn    | defining a method named with an established prefix (like "into_") that takes `self` with the wrong convention
[wrong_transmute](https://github.com/rust-lang-nursery/rust-clippy/wiki#wrong_transmute)                                     | warn    | transmutes that are confusing at best, undefined behaviour at worst and always useless
[zero_divided_by_zero](https://github.com/rust-lang-nursery/rust-clippy/wiki#zero_divided_by_zero)                           | warn    | usage of `0.0 / 0.0` to obtain NaN instead of std::f32::NaN or std::f64::NaN
[zero_prefixed_literal](https://github.com/rust-lang-nursery/rust-clippy/wiki#zero_prefixed_literal)                         | warn    | integer literals starting with `0`
[zero_ptr](https://github.com/rust-lang-nursery/rust-clippy/wiki#zero_ptr)                                                   | warn    | using 0 as *{const, mut} T
[zero_width_space](https://github.com/rust-lang-nursery/rust-clippy/wiki#zero_width_space)                                   | deny    | using a zero-width space in a string literal, which is confusing

More to come, please [file an issue](https://github.com/rust-lang-nursery/rust-clippy/issues) if you have ideas!

## License

Licensed under [MPL](https://www.mozilla.org/MPL/2.0/).
If you're having issues with the license, let me know and I'll try to change it to something more permissive.
