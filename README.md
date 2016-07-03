# rust-clippy

[![Build Status](https://travis-ci.org/Manishearth/rust-clippy.svg?branch=master)](https://travis-ci.org/Manishearth/rust-clippy)
[![Clippy Linting Result](http://clippy.bashy.io/github/Manishearth/rust-clippy/master/badge.svg)](http://clippy.bashy.io/github/Manishearth/rust-clippy/master/log)
[![Current Version](http://meritbadge.herokuapp.com/clippy)](https://crates.io/crates/clippy)
[![License: MPL-2.0](https://img.shields.io/crates/l/clippy.svg)](#License)

A collection of lints to catch common mistakes and improve your Rust code.

Table of contents:

*   [Lint list](#lints)
*   [Usage instructions](#usage)
*   [Configuration](#configuration)
*   [*clippy-service*](#link-with-clippy-service)
*   [License](#license)

## Lints

There are 158 lints included in this crate:

name                                                                                                                 | default | meaning
---------------------------------------------------------------------------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
[absurd_extreme_comparisons](https://github.com/Manishearth/rust-clippy/wiki#absurd_extreme_comparisons)             | warn    | a comparison involving a maximum or minimum value involves a case that is always true or always false
[almost_swapped](https://github.com/Manishearth/rust-clippy/wiki#almost_swapped)                                     | warn    | `foo = bar; bar = foo` sequence
[approx_constant](https://github.com/Manishearth/rust-clippy/wiki#approx_constant)                                   | warn    | the approximate of a known float constant (in `std::f64::consts` or `std::f32::consts`) is found; suggests to use the constant
[assign_op_pattern](https://github.com/Manishearth/rust-clippy/wiki#assign_op_pattern)                               | warn    | assigning the result of an operation on a variable to that same variable
[assign_ops](https://github.com/Manishearth/rust-clippy/wiki#assign_ops)                                             | allow   | any assignment operation
[bad_bit_mask](https://github.com/Manishearth/rust-clippy/wiki#bad_bit_mask)                                         | warn    | expressions of the form `_ & mask == select` that will only ever return `true` or `false` (because in the example `select` containing bits that `mask` doesn't have)
[blacklisted_name](https://github.com/Manishearth/rust-clippy/wiki#blacklisted_name)                                 | warn    | usage of a blacklisted/placeholder name
[block_in_if_condition_expr](https://github.com/Manishearth/rust-clippy/wiki#block_in_if_condition_expr)             | warn    | braces can be eliminated in conditions that are expressions, e.g `if { true } ...`
[block_in_if_condition_stmt](https://github.com/Manishearth/rust-clippy/wiki#block_in_if_condition_stmt)             | warn    | avoid complex blocks in conditions, instead move the block higher and bind it with 'let'; e.g: `if { let x = true; x } ...`
[bool_comparison](https://github.com/Manishearth/rust-clippy/wiki#bool_comparison)                                   | warn    | comparing a variable to a boolean, e.g. `if x == true`
[box_vec](https://github.com/Manishearth/rust-clippy/wiki#box_vec)                                                   | warn    | usage of `Box<Vec<T>>`, vector elements are already on the heap
[boxed_local](https://github.com/Manishearth/rust-clippy/wiki#boxed_local)                                           | warn    | using `Box<T>` where unnecessary
[cast_possible_truncation](https://github.com/Manishearth/rust-clippy/wiki#cast_possible_truncation)                 | allow   | casts that may cause truncation of the value, e.g `x as u8` where `x: u32`, or `x as i32` where `x: f32`
[cast_possible_wrap](https://github.com/Manishearth/rust-clippy/wiki#cast_possible_wrap)                             | allow   | casts that may cause wrapping around the value, e.g `x as i32` where `x: u32` and `x > i32::MAX`
[cast_precision_loss](https://github.com/Manishearth/rust-clippy/wiki#cast_precision_loss)                           | allow   | casts that cause loss of precision, e.g `x as f32` where `x: u64`
[cast_sign_loss](https://github.com/Manishearth/rust-clippy/wiki#cast_sign_loss)                                     | allow   | casts from signed types to unsigned types, e.g `x as u32` where `x: i32`
[char_lit_as_u8](https://github.com/Manishearth/rust-clippy/wiki#char_lit_as_u8)                                     | warn    | Casting a character literal to u8
[chars_next_cmp](https://github.com/Manishearth/rust-clippy/wiki#chars_next_cmp)                                     | warn    | using `.chars().next()` to check if a string starts with a char
[clone_double_ref](https://github.com/Manishearth/rust-clippy/wiki#clone_double_ref)                                 | warn    | using `clone` on `&&T`
[clone_on_copy](https://github.com/Manishearth/rust-clippy/wiki#clone_on_copy)                                       | warn    | using `clone` on a `Copy` type
[cmp_nan](https://github.com/Manishearth/rust-clippy/wiki#cmp_nan)                                                   | deny    | comparisons to NAN (which will always return false, which is probably not intended)
[cmp_owned](https://github.com/Manishearth/rust-clippy/wiki#cmp_owned)                                               | warn    | creating owned instances for comparing with others, e.g. `x == "foo".to_string()`
[collapsible_if](https://github.com/Manishearth/rust-clippy/wiki#collapsible_if)                                     | warn    | two nested `if`-expressions can be collapsed into one, e.g. `if x { if y { foo() } }` can be written as `if x && y { foo() }` and an `else { if .. } expression can be collapsed to `else if`
[crosspointer_transmute](https://github.com/Manishearth/rust-clippy/wiki#crosspointer_transmute)                     | warn    | transmutes that have to or from types that are a pointer to the other
[cyclomatic_complexity](https://github.com/Manishearth/rust-clippy/wiki#cyclomatic_complexity)                       | warn    | finds functions that should be split up into multiple functions
[deprecated_semver](https://github.com/Manishearth/rust-clippy/wiki#deprecated_semver)                               | warn    | `Warn` on `#[deprecated(since = "x")]` where x is not semver
[derive_hash_xor_eq](https://github.com/Manishearth/rust-clippy/wiki#derive_hash_xor_eq)                             | warn    | deriving `Hash` but implementing `PartialEq` explicitly
[doc_markdown](https://github.com/Manishearth/rust-clippy/wiki#doc_markdown)                                         | warn    | checks for the presence of `_`, `::` or camel-case outside ticks in documentation
[double_neg](https://github.com/Manishearth/rust-clippy/wiki#double_neg)                                             | warn    | `--x` is a double negation of `x` and not a pre-decrement as in C or C++
[drop_ref](https://github.com/Manishearth/rust-clippy/wiki#drop_ref)                                                 | warn    | call to `std::mem::drop` with a reference instead of an owned value, which will not call the `Drop::drop` method on the underlying value
[duplicate_underscore_argument](https://github.com/Manishearth/rust-clippy/wiki#duplicate_underscore_argument)       | warn    | Function arguments having names which only differ by an underscore
[empty_loop](https://github.com/Manishearth/rust-clippy/wiki#empty_loop)                                             | warn    | empty `loop {}` detected
[enum_clike_unportable_variant](https://github.com/Manishearth/rust-clippy/wiki#enum_clike_unportable_variant)       | warn    | finds C-like enums that are `repr(isize/usize)` and have values that don't fit into an `i32`
[enum_glob_use](https://github.com/Manishearth/rust-clippy/wiki#enum_glob_use)                                       | allow   | finds use items that import all variants of an enum
[enum_variant_names](https://github.com/Manishearth/rust-clippy/wiki#enum_variant_names)                             | warn    | finds enums where all variants share a prefix/postfix
[eq_op](https://github.com/Manishearth/rust-clippy/wiki#eq_op)                                                       | warn    | equal operands on both sides of a comparison or bitwise combination (e.g. `x == x`)
[expl_impl_clone_on_copy](https://github.com/Manishearth/rust-clippy/wiki#expl_impl_clone_on_copy)                   | warn    | implementing `Clone` explicitly on `Copy` types
[explicit_counter_loop](https://github.com/Manishearth/rust-clippy/wiki#explicit_counter_loop)                       | warn    | for-looping with an explicit counter when `_.enumerate()` would do
[explicit_iter_loop](https://github.com/Manishearth/rust-clippy/wiki#explicit_iter_loop)                             | warn    | for-looping over `_.iter()` or `_.iter_mut()` when `&_` or `&mut _` would do
[extend_from_slice](https://github.com/Manishearth/rust-clippy/wiki#extend_from_slice)                               | warn    | `.extend_from_slice(_)` is a faster way to extend a Vec by a slice
[filter_map](https://github.com/Manishearth/rust-clippy/wiki#filter_map)                                             | allow   | using combinations of `filter`, `map`, `filter_map` and `flat_map` which can usually be written as a single method call
[filter_next](https://github.com/Manishearth/rust-clippy/wiki#filter_next)                                           | warn    | using `filter(p).next()`, which is more succinctly expressed as `.find(p)`
[float_arithmetic](https://github.com/Manishearth/rust-clippy/wiki#float_arithmetic)                                 | allow   | Any floating-point arithmetic statement
[float_cmp](https://github.com/Manishearth/rust-clippy/wiki#float_cmp)                                               | warn    | using `==` or `!=` on float values (as floating-point operations usually involve rounding errors, it is always better to check for approximate equality within small bounds)
[for_kv_map](https://github.com/Manishearth/rust-clippy/wiki#for_kv_map)                                             | warn    | looping on a map using `iter` when `keys` or `values` would do
[for_loop_over_option](https://github.com/Manishearth/rust-clippy/wiki#for_loop_over_option)                         | warn    | for-looping over an `Option`, which is more clearly expressed as an `if let`
[for_loop_over_result](https://github.com/Manishearth/rust-clippy/wiki#for_loop_over_result)                         | warn    | for-looping over a `Result`, which is more clearly expressed as an `if let`
[identity_op](https://github.com/Manishearth/rust-clippy/wiki#identity_op)                                           | warn    | using identity operations, e.g. `x + 0` or `y / 1`
[if_not_else](https://github.com/Manishearth/rust-clippy/wiki#if_not_else)                                           | allow   | finds if branches that could be swapped so no negation operation is necessary on the condition
[if_same_then_else](https://github.com/Manishearth/rust-clippy/wiki#if_same_then_else)                               | warn    | if with the same *then* and *else* blocks
[ifs_same_cond](https://github.com/Manishearth/rust-clippy/wiki#ifs_same_cond)                                       | warn    | consecutive `ifs` with the same condition
[indexing_slicing](https://github.com/Manishearth/rust-clippy/wiki#indexing_slicing)                                 | allow   | indexing/slicing usage
[ineffective_bit_mask](https://github.com/Manishearth/rust-clippy/wiki#ineffective_bit_mask)                         | warn    | expressions where a bit mask will be rendered useless by a comparison, e.g. `(x | 1) > 2`
[inline_always](https://github.com/Manishearth/rust-clippy/wiki#inline_always)                                       | warn    | `#[inline(always)]` is a bad idea in most cases
[integer_arithmetic](https://github.com/Manishearth/rust-clippy/wiki#integer_arithmetic)                             | allow   | Any integer arithmetic statement
[invalid_regex](https://github.com/Manishearth/rust-clippy/wiki#invalid_regex)                                       | deny    | finds invalid regular expressions
[invalid_upcast_comparisons](https://github.com/Manishearth/rust-clippy/wiki#invalid_upcast_comparisons)             | allow   | a comparison involving an upcast which is always true or false
[items_after_statements](https://github.com/Manishearth/rust-clippy/wiki#items_after_statements)                     | allow   | finds blocks where an item comes after a statement
[iter_next_loop](https://github.com/Manishearth/rust-clippy/wiki#iter_next_loop)                                     | warn    | for-looping over `_.next()` which is probably not intended
[iter_nth](https://github.com/Manishearth/rust-clippy/wiki#iter_nth)                                                 | warn    | using `.iter().nth()` on a standard library type with O(1) element access
[len_without_is_empty](https://github.com/Manishearth/rust-clippy/wiki#len_without_is_empty)                         | warn    | traits and impls that have `.len()` but not `.is_empty()`
[len_zero](https://github.com/Manishearth/rust-clippy/wiki#len_zero)                                                 | warn    | checking `.len() == 0` or `.len() > 0` (or similar) when `.is_empty()` could be used instead
[let_and_return](https://github.com/Manishearth/rust-clippy/wiki#let_and_return)                                     | warn    | creating a let-binding and then immediately returning it like `let x = expr; x` at the end of a block
[let_unit_value](https://github.com/Manishearth/rust-clippy/wiki#let_unit_value)                                     | warn    | creating a let binding to a value of unit type, which usually can't be used afterwards
[linkedlist](https://github.com/Manishearth/rust-clippy/wiki#linkedlist)                                             | warn    | usage of LinkedList, usually a vector is faster, or a more specialized data structure like a VecDeque
[logic_bug](https://github.com/Manishearth/rust-clippy/wiki#logic_bug)                                               | warn    | checks for boolean expressions that contain terminals which can be eliminated
[manual_swap](https://github.com/Manishearth/rust-clippy/wiki#manual_swap)                                           | warn    | manual swap
[many_single_char_names](https://github.com/Manishearth/rust-clippy/wiki#many_single_char_names)                     | warn    | too many single character bindings
[map_clone](https://github.com/Manishearth/rust-clippy/wiki#map_clone)                                               | warn    | using `.map(|x| x.clone())` to clone an iterator or option's contents (recommends `.cloned()` instead)
[map_entry](https://github.com/Manishearth/rust-clippy/wiki#map_entry)                                               | warn    | use of `contains_key` followed by `insert` on a `HashMap` or `BTreeMap`
[match_bool](https://github.com/Manishearth/rust-clippy/wiki#match_bool)                                             | warn    | a match on boolean expression; recommends `if..else` block instead
[match_overlapping_arm](https://github.com/Manishearth/rust-clippy/wiki#match_overlapping_arm)                       | warn    | a match has overlapping arms
[match_ref_pats](https://github.com/Manishearth/rust-clippy/wiki#match_ref_pats)                                     | warn    | a match or `if let` has all arms prefixed with `&`; the match expression can be dereferenced instead
[match_same_arms](https://github.com/Manishearth/rust-clippy/wiki#match_same_arms)                                   | warn    | `match` with identical arm bodies
[mem_forget](https://github.com/Manishearth/rust-clippy/wiki#mem_forget)                                             | allow   | `mem::forget` usage on `Drop` types is likely to cause memory leaks
[min_max](https://github.com/Manishearth/rust-clippy/wiki#min_max)                                                   | warn    | `min(_, max(_, _))` (or vice versa) with bounds clamping the result to a constant
[modulo_one](https://github.com/Manishearth/rust-clippy/wiki#modulo_one)                                             | warn    | taking a number modulo 1, which always returns 0
[mut_mut](https://github.com/Manishearth/rust-clippy/wiki#mut_mut)                                                   | allow   | usage of double-mut refs, e.g. `&mut &mut ...` (either copy'n'paste error, or shows a fundamental misunderstanding of references)
[mutex_atomic](https://github.com/Manishearth/rust-clippy/wiki#mutex_atomic)                                         | warn    | using a Mutex where an atomic value could be used instead
[mutex_integer](https://github.com/Manishearth/rust-clippy/wiki#mutex_integer)                                       | allow   | using a Mutex for an integer type
[needless_bool](https://github.com/Manishearth/rust-clippy/wiki#needless_bool)                                       | warn    | if-statements with plain booleans in the then- and else-clause, e.g. `if p { true } else { false }`
[needless_borrow](https://github.com/Manishearth/rust-clippy/wiki#needless_borrow)                                   | warn    | taking a reference that is going to be automatically dereferenced
[needless_lifetimes](https://github.com/Manishearth/rust-clippy/wiki#needless_lifetimes)                             | warn    | using explicit lifetimes for references in function arguments when elision rules would allow omitting them
[needless_range_loop](https://github.com/Manishearth/rust-clippy/wiki#needless_range_loop)                           | warn    | for-looping over a range of indices where an iterator over items would do
[needless_return](https://github.com/Manishearth/rust-clippy/wiki#needless_return)                                   | warn    | using a return statement like `return expr;` where an expression would suffice
[needless_update](https://github.com/Manishearth/rust-clippy/wiki#needless_update)                                   | warn    | using `{ ..base }` when there are no missing fields
[neg_multiply](https://github.com/Manishearth/rust-clippy/wiki#neg_multiply)                                         | warn    | Warns on multiplying integers with -1
[new_ret_no_self](https://github.com/Manishearth/rust-clippy/wiki#new_ret_no_self)                                   | warn    | not returning `Self` in a `new` method
[new_without_default](https://github.com/Manishearth/rust-clippy/wiki#new_without_default)                           | warn    | `fn new() -> Self` method without `Default` implementation
[new_without_default_derive](https://github.com/Manishearth/rust-clippy/wiki#new_without_default_derive)             | warn    | `fn new() -> Self` without `#[derive]`able `Default` implementation
[no_effect](https://github.com/Manishearth/rust-clippy/wiki#no_effect)                                               | warn    | statements with no effect
[non_ascii_literal](https://github.com/Manishearth/rust-clippy/wiki#non_ascii_literal)                               | allow   | using any literal non-ASCII chars in a string literal; suggests using the \\u escape instead
[nonminimal_bool](https://github.com/Manishearth/rust-clippy/wiki#nonminimal_bool)                                   | allow   | checks for boolean expressions that can be written more concisely
[nonsensical_open_options](https://github.com/Manishearth/rust-clippy/wiki#nonsensical_open_options)                 | warn    | nonsensical combination of options for opening a file
[not_unsafe_ptr_arg_deref](https://github.com/Manishearth/rust-clippy/wiki#not_unsafe_ptr_arg_deref)                 | warn    | public functions dereferencing raw pointer arguments but not marked `unsafe`
[ok_expect](https://github.com/Manishearth/rust-clippy/wiki#ok_expect)                                               | warn    | using `ok().expect()`, which gives worse error messages than calling `expect` directly on the Result
[option_map_unwrap_or](https://github.com/Manishearth/rust-clippy/wiki#option_map_unwrap_or)                         | warn    | using `Option.map(f).unwrap_or(a)`, which is more succinctly expressed as `map_or(a, f)`
[option_map_unwrap_or_else](https://github.com/Manishearth/rust-clippy/wiki#option_map_unwrap_or_else)               | warn    | using `Option.map(f).unwrap_or_else(g)`, which is more succinctly expressed as `map_or_else(g, f)`
[option_unwrap_used](https://github.com/Manishearth/rust-clippy/wiki#option_unwrap_used)                             | allow   | using `Option.unwrap()`, which should at least get a better message using `expect()`
[or_fun_call](https://github.com/Manishearth/rust-clippy/wiki#or_fun_call)                                           | warn    | using any `*or` method when the `*or_else` would do
[out_of_bounds_indexing](https://github.com/Manishearth/rust-clippy/wiki#out_of_bounds_indexing)                     | deny    | out of bound constant indexing
[overflow_check_conditional](https://github.com/Manishearth/rust-clippy/wiki#overflow_check_conditional)             | warn    | Using overflow checks which are likely to panic
[panic_params](https://github.com/Manishearth/rust-clippy/wiki#panic_params)                                         | warn    | missing parameters in `panic!`
[precedence](https://github.com/Manishearth/rust-clippy/wiki#precedence)                                             | warn    | catches operations where precedence may be unclear. See the wiki for a list of cases caught
[print_stdout](https://github.com/Manishearth/rust-clippy/wiki#print_stdout)                                         | allow   | printing on stdout
[ptr_arg](https://github.com/Manishearth/rust-clippy/wiki#ptr_arg)                                                   | warn    | fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` instead, respectively
[range_step_by_zero](https://github.com/Manishearth/rust-clippy/wiki#range_step_by_zero)                             | warn    | using Range::step_by(0), which produces an infinite iterator
[range_zip_with_len](https://github.com/Manishearth/rust-clippy/wiki#range_zip_with_len)                             | warn    | zipping iterator with a range when enumerate() would do
[redundant_closure](https://github.com/Manishearth/rust-clippy/wiki#redundant_closure)                               | warn    | using redundant closures, i.e. `|a| foo(a)` (which can be written as just `foo`)
[redundant_closure_call](https://github.com/Manishearth/rust-clippy/wiki#redundant_closure_call)                     | warn    | Closures should not be called in the expression they are defined
[redundant_pattern](https://github.com/Manishearth/rust-clippy/wiki#redundant_pattern)                               | warn    | using `name @ _` in a pattern
[regex_macro](https://github.com/Manishearth/rust-clippy/wiki#regex_macro)                                           | warn    | finds use of `regex!(_)`, suggests `Regex::new(_)` instead
[result_unwrap_used](https://github.com/Manishearth/rust-clippy/wiki#result_unwrap_used)                             | allow   | using `Result.unwrap()`, which might be better handled
[reverse_range_loop](https://github.com/Manishearth/rust-clippy/wiki#reverse_range_loop)                             | warn    | Iterating over an empty range, such as `10..0` or `5..5`
[search_is_some](https://github.com/Manishearth/rust-clippy/wiki#search_is_some)                                     | warn    | using an iterator search followed by `is_some()`, which is more succinctly expressed as a call to `any()`
[shadow_reuse](https://github.com/Manishearth/rust-clippy/wiki#shadow_reuse)                                         | allow   | rebinding a name to an expression that re-uses the original value, e.g. `let x = x + 1`
[shadow_same](https://github.com/Manishearth/rust-clippy/wiki#shadow_same)                                           | allow   | rebinding a name to itself, e.g. `let mut x = &mut x`
[shadow_unrelated](https://github.com/Manishearth/rust-clippy/wiki#shadow_unrelated)                                 | allow   | The name is re-bound without even using the original value
[should_implement_trait](https://github.com/Manishearth/rust-clippy/wiki#should_implement_trait)                     | warn    | defining a method that should be implementing a std trait
[similar_names](https://github.com/Manishearth/rust-clippy/wiki#similar_names)                                       | allow   | similarly named items and bindings
[single_char_pattern](https://github.com/Manishearth/rust-clippy/wiki#single_char_pattern)                           | warn    | using a single-character str where a char could be used, e.g. `_.split("x")`
[single_match](https://github.com/Manishearth/rust-clippy/wiki#single_match)                                         | warn    | a match statement with a single nontrivial arm (i.e, where the other arm is `_ => {}`) is used; recommends `if let` instead
[single_match_else](https://github.com/Manishearth/rust-clippy/wiki#single_match_else)                               | allow   | a match statement with a two arms where the second arm's pattern is a wildcard; recommends `if let` instead
[string_add](https://github.com/Manishearth/rust-clippy/wiki#string_add)                                             | allow   | using `x + ..` where x is a `String`; suggests using `push_str()` instead
[string_add_assign](https://github.com/Manishearth/rust-clippy/wiki#string_add_assign)                               | allow   | using `x = x + ..` where x is a `String`; suggests using `push_str()` instead
[string_lit_as_bytes](https://github.com/Manishearth/rust-clippy/wiki#string_lit_as_bytes)                           | warn    | calling `as_bytes` on a string literal; suggests using a byte string literal instead
[stutter](https://github.com/Manishearth/rust-clippy/wiki#stutter)                                                   | allow   | finds type names prefixed/postfixed with their containing module's name
[suspicious_assignment_formatting](https://github.com/Manishearth/rust-clippy/wiki#suspicious_assignment_formatting) | warn    | suspicious formatting of `*=`, `-=` or `!=`
[suspicious_else_formatting](https://github.com/Manishearth/rust-clippy/wiki#suspicious_else_formatting)             | warn    | suspicious formatting of `else if`
[temporary_assignment](https://github.com/Manishearth/rust-clippy/wiki#temporary_assignment)                         | warn    | assignments to temporaries
[temporary_cstring_as_ptr](https://github.com/Manishearth/rust-clippy/wiki#temporary_cstring_as_ptr)                 | warn    | getting the inner pointer of a temporary `CString`
[too_many_arguments](https://github.com/Manishearth/rust-clippy/wiki#too_many_arguments)                             | warn    | functions with too many arguments
[toplevel_ref_arg](https://github.com/Manishearth/rust-clippy/wiki#toplevel_ref_arg)                                 | warn    | An entire binding was declared as `ref`, in a function argument (`fn foo(ref x: Bar)`), or a `let` statement (`let ref x = foo()`). In such cases, it is preferred to take references with `&`.
[transmute_ptr_to_ref](https://github.com/Manishearth/rust-clippy/wiki#transmute_ptr_to_ref)                         | warn    | transmutes from a pointer to a reference type
[trivial_regex](https://github.com/Manishearth/rust-clippy/wiki#trivial_regex)                                       | warn    | finds trivial regular expressions
[type_complexity](https://github.com/Manishearth/rust-clippy/wiki#type_complexity)                                   | warn    | usage of very complex types; recommends factoring out parts into `type` definitions
[unicode_not_nfc](https://github.com/Manishearth/rust-clippy/wiki#unicode_not_nfc)                                   | allow   | using a unicode literal not in NFC normal form (see [unicode tr15](http://www.unicode.org/reports/tr15/) for further information)
[unit_cmp](https://github.com/Manishearth/rust-clippy/wiki#unit_cmp)                                                 | warn    | comparing unit values (which is always `true` or `false`, respectively)
[unnecessary_mut_passed](https://github.com/Manishearth/rust-clippy/wiki#unnecessary_mut_passed)                     | warn    | an argument is passed as a mutable reference although the function/method only demands an immutable reference
[unnecessary_operation](https://github.com/Manishearth/rust-clippy/wiki#unnecessary_operation)                       | warn    | outer expressions with no effect
[unneeded_field_pattern](https://github.com/Manishearth/rust-clippy/wiki#unneeded_field_pattern)                     | warn    | Struct fields are bound to a wildcard instead of using `..`
[unsafe_removed_from_name](https://github.com/Manishearth/rust-clippy/wiki#unsafe_removed_from_name)                 | warn    | unsafe removed from name
[unused_collect](https://github.com/Manishearth/rust-clippy/wiki#unused_collect)                                     | warn    | `collect()`ing an iterator without using the result; this is usually better written as a for loop
[unused_label](https://github.com/Manishearth/rust-clippy/wiki#unused_label)                                         | warn    | unused label
[unused_lifetimes](https://github.com/Manishearth/rust-clippy/wiki#unused_lifetimes)                                 | warn    | unused lifetimes in function definitions
[use_debug](https://github.com/Manishearth/rust-clippy/wiki#use_debug)                                               | allow   | use `Debug`-based formatting
[used_underscore_binding](https://github.com/Manishearth/rust-clippy/wiki#used_underscore_binding)                   | allow   | using a binding which is prefixed with an underscore
[useless_format](https://github.com/Manishearth/rust-clippy/wiki#useless_format)                                     | warn    | useless use of `format!`
[useless_let_if_seq](https://github.com/Manishearth/rust-clippy/wiki#useless_let_if_seq)                             | warn    | Checks for unidiomatic `let mut` declaration followed by initialization in `if`
[useless_transmute](https://github.com/Manishearth/rust-clippy/wiki#useless_transmute)                               | warn    | transmutes that have the same to and from types or could be a cast/coercion
[useless_vec](https://github.com/Manishearth/rust-clippy/wiki#useless_vec)                                           | warn    | useless `vec!`
[while_let_loop](https://github.com/Manishearth/rust-clippy/wiki#while_let_loop)                                     | warn    | `loop { if let { ... } else break }` can be written as a `while let` loop
[while_let_on_iterator](https://github.com/Manishearth/rust-clippy/wiki#while_let_on_iterator)                       | warn    | using a while-let loop instead of a for loop on an iterator
[wrong_pub_self_convention](https://github.com/Manishearth/rust-clippy/wiki#wrong_pub_self_convention)               | allow   | defining a public method named with an established prefix (like "into_") that takes `self` with the wrong convention
[wrong_self_convention](https://github.com/Manishearth/rust-clippy/wiki#wrong_self_convention)                       | warn    | defining a method named with an established prefix (like "into_") that takes `self` with the wrong convention
[wrong_transmute](https://github.com/Manishearth/rust-clippy/wiki#wrong_transmute)                                   | warn    | transmutes that are confusing at best, undefined behaviour at worst and always useless
[zero_divided_by_zero](https://github.com/Manishearth/rust-clippy/wiki#zero_divided_by_zero)                         | warn    | usage of `0.0 / 0.0` to obtain NaN instead of std::f32::NaN or std::f64::NaN
[zero_width_space](https://github.com/Manishearth/rust-clippy/wiki#zero_width_space)                                 | deny    | using a zero-width space in a string literal, which is confusing

More to come, please [file an issue](https://github.com/Manishearth/rust-clippy/issues) if you have ideas!

## Usage

As a general rule clippy will only work with the *latest* Rust nightly for now.

### As a Compiler Plugin

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

### As a cargo subcommand (`cargo clippy`)

An alternate way to use clippy is by installing clippy through cargo as a cargo
subcommand.

```terminal
cargo install clippy
```

Now you can run clippy by invoking `cargo clippy`, or
`multirust run nightly cargo clippy` directly from a directory that is usually
compiled with stable.

In case you are not using multirust, you need to set the environment flag
`SYSROOT` during installation so clippy knows where to find `librustc` and
similar crates.

```terminal
SYSROOT=/path/to/rustc/sysroot cargo install clippy
```

### Running clippy from the command line without installing

To have cargo compile your crate with clippy without needing `#![plugin(clippy)]`
in your code, you can use:

```terminal
cargo rustc -- -L /path/to/clippy_so -Z extra-plugins=clippy
```

*[Note](https://github.com/Manishearth/rust-clippy/wiki#a-word-of-warning):*
Be sure that clippy was compiled with the same version of rustc that cargo invokes here!

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

Instead of adding the `cfg_attr` attributes you can also run clippy on demand:
`cargo rustc --features clippy -- -Z no-trans -Z extra-plugins=clippy`
(the `-Z no trans`, while not neccessary, will stop the compilation process after
typechecking (and lints) have completed, which can significantly reduce the runtime).

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

## Link with clippy service

`clippy-service` is a rust web initiative providing `rust-clippy` as a web service.

Both projects are independent and maintained by different people
(even if some `clippy-service`'s contributions are authored by some `rust-clippy` members).

You can check out this great service at [clippy.bashy.io](https://clippy.bashy.io/).

## License

Licensed under [MPL](https://www.mozilla.org/MPL/2.0/).
If you're having issues with the license, let me know and I'll try to change it to something more permissive.
