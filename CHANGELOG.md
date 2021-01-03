# Changelog

All notable changes to this project will be documented in this file.
See [Changelog Update](doc/changelog_update.md) if you want to update this
document.

## Unreleased / In Rust Nightly

[4911ab1...master](https://github.com/rust-lang/rust-clippy/compare/4911ab1...master)

## Rust 1.50

Current beta, release 2021-02-11

[b20d4c1...4911ab1](https://github.com/rust-lang/rust-clippy/compare/b20d4c1...4911ab1)

### New Lints

* [`suspicious_operation_groupings`] [#6086](https://github.com/rust-lang/rust-clippy/pull/6086)
* [`size_of_in_element_count`] [#6394](https://github.com/rust-lang/rust-clippy/pull/6394)
* [`unnecessary_wraps`] [#6070](https://github.com/rust-lang/rust-clippy/pull/6070)
* [`let_underscore_drop`] [#6305](https://github.com/rust-lang/rust-clippy/pull/6305)
* [`collapsible_match`] [#6402](https://github.com/rust-lang/rust-clippy/pull/6402)
* [`redundant_else`] [#6330](https://github.com/rust-lang/rust-clippy/pull/6330)
* [`zero_sized_map_values`] [#6218](https://github.com/rust-lang/rust-clippy/pull/6218)
* [`print_stderr`] [#6367](https://github.com/rust-lang/rust-clippy/pull/6367)
* [`string_from_utf8_as_bytes`] [#6134](https://github.com/rust-lang/rust-clippy/pull/6134)

### Moves and Deprecations

* Previously deprecated [`str_to_string`] and [`string_to_string`] have been un-deprecated
  as `restriction` lints [#6333](https://github.com/rust-lang/rust-clippy/pull/6333)
* Deprecate [`panic_params`] lint. This is now available in rustc as `panic_fmt`
  [#6351](https://github.com/rust-lang/rust-clippy/pull/6351)
* Move [`map_err_ignore`] to `restriction`
  [#6416](https://github.com/rust-lang/rust-clippy/pull/6416)
* Move [`await_holding_refcell_ref`] to `pedantic`
  [#6354](https://github.com/rust-lang/rust-clippy/pull/6354)
* Move [`await_holding_lock`] to `pedantic`
  [#6354](https://github.com/rust-lang/rust-clippy/pull/6354)

### Enhancements

* Add the `unreadable-literal-lint-fractions` configuration to disable
  the `unreadable_literal` lint for fractions
  [#6421](https://github.com/rust-lang/rust-clippy/pull/6421)
* [`clone_on_copy`]: Now shows the type in the lint message
  [#6443](https://github.com/rust-lang/rust-clippy/pull/6443)
* [`redundant_pattern_matching`]: Now also lints on `std::task::Poll`
  [#6339](https://github.com/rust-lang/rust-clippy/pull/6339)
* [`redundant_pattern_matching`]: Additionally also lints on `std::net::IpAddr`
  [#6377](https://github.com/rust-lang/rust-clippy/pull/6377)
* [`search_is_some`]: Now suggests `contains` instead of `find(foo).is_some()`
  [#6119](https://github.com/rust-lang/rust-clippy/pull/6119)
* [`clone_double_ref`]: Now prints the reference type in the lint message
  [#6442](https://github.com/rust-lang/rust-clippy/pull/6442)
* [`modulo_one`]: Now also lints on -1.
  [#6360](https://github.com/rust-lang/rust-clippy/pull/6360)
* [`empty_loop`]: Now lints no_std crates, too
  [#6205](https://github.com/rust-lang/rust-clippy/pull/6205)
* [`or_fun_call`]: Now also lints when indexing `HashMap` or `BTreeMap`
  [#6267](https://github.com/rust-lang/rust-clippy/pull/6267)
* [`wrong_self_convention`]: Now also lints in trait definitions
  [#6316](https://github.com/rust-lang/rust-clippy/pull/6316)
* [`needless_borrow`]: Print the type in the lint message
  [#6449](https://github.com/rust-lang/rust-clippy/pull/6449)

[msrv_readme]: https://github.com/rust-lang/rust-clippy#specifying-the-minimum-supported-rust-version

### False Positive Fixes

* [`manual_range_contains`]: No longer lints in `const fn`
  [#6382](https://github.com/rust-lang/rust-clippy/pull/6382)
* [`unnecessary_lazy_evaluations`]: No longer lints if closure argument is used
  [#6370](https://github.com/rust-lang/rust-clippy/pull/6370)
* [`match_single_binding`]: Now ignores cases with `#[cfg()]` macros
  [#6435](https://github.com/rust-lang/rust-clippy/pull/6435)
* [`match_like_matches_macro`]: No longer lints on arms with attributes
  [#6290](https://github.com/rust-lang/rust-clippy/pull/6290)
* [`map_clone`]: No longer lints with deref and clone
  [#6269](https://github.com/rust-lang/rust-clippy/pull/6269)
* [`map_clone`]: No longer lints in the case of &mut
  [#6301](https://github.com/rust-lang/rust-clippy/pull/6301)
* [`needless_update`]: Now ignores `non_exhaustive` structs
  [#6464](https://github.com/rust-lang/rust-clippy/pull/6464)
* [`needless_collect`]: No longer lints when a collect is needed multiple times
  [#6313](https://github.com/rust-lang/rust-clippy/pull/6313)
* [`unnecessary_cast`] No longer lints cfg-dependent types
  [#6369](https://github.com/rust-lang/rust-clippy/pull/6369)
* [`declare_interior_mutable_const`] and [`borrow_interior_mutable_const`]:
  Both now ignore enums with frozen variants
  [#6110](https://github.com/rust-lang/rust-clippy/pull/6110)


### Suggestion Fixes/Improvements

* [`vec_box`]: Provide correct type scope suggestion
  [#6271](https://github.com/rust-lang/rust-clippy/pull/6271)
* [`manual_range_contains`]: Give correct suggestion when using floats
  [#6320](https://github.com/rust-lang/rust-clippy/pull/6320)
* [`unnecessary_lazy_evaluations`]: Don't always mark suggestion as MachineApplicable
  [#6272](https://github.com/rust-lang/rust-clippy/pull/6272)
* [`manual_async_fn`]: Improve suggestion formatting
  [#6294](https://github.com/rust-lang/rust-clippy/pull/6294)
* [`unnecessary_cast`]: Fix incorrectly formatted float literal suggestion
  [#6362](https://github.com/rust-lang/rust-clippy/pull/6362)

### ICE Fixes

* Fix a crash in [`from_iter_instead_of_collect`]
  [#6304](https://github.com/rust-lang/rust-clippy/pull/6304)
* Fix a silent crash when parsing doc comments in [`needless_doctest_main`]
  [#6458](https://github.com/rust-lang/rust-clippy/pull/6458)

### Documentation Improvements

* The lint website search has been improved ([#6477](https://github.com/rust-lang/rust-clippy/pull/6477)):
  * Searching for lints with dashes and spaces is possible now. For example
    `missing-errors-doc` and `missing errors doc` are now valid aliases for lint names
  * Improved fuzzy search in lint descriptions
* Various README improvements
  [#6287](https://github.com/rust-lang/rust-clippy/pull/6287)
* Add known problems to [`comparison_chain`] documentation
  [#6390](https://github.com/rust-lang/rust-clippy/pull/6390)
* Fix example used in [`cargo_common_metadata`]
  [#6293](https://github.com/rust-lang/rust-clippy/pull/6293)
* Improve [`map_clone`] documentation
  [#6340](https://github.com/rust-lang/rust-clippy/pull/6340)

### Others

* You can now tell Clippy about the MSRV your project supports. Please refer to
  the specific README section to learn more about MSRV support [here][msrv_readme]
  [#6201](https://github.com/rust-lang/rust-clippy/pull/6201)
* Add `--no-deps` option to avoid running on path dependencies in workspaces
  [#6188](https://github.com/rust-lang/rust-clippy/pull/6188)

## Rust 1.49

Current stable, released 2020-12-31

[e636b88...b20d4c1](https://github.com/rust-lang/rust-clippy/compare/e636b88...b20d4c1)

### New Lints

* [`field_reassign_with_default`] [#5911](https://github.com/rust-lang/rust-clippy/pull/5911)
* [`await_holding_refcell_ref`] [#6029](https://github.com/rust-lang/rust-clippy/pull/6029)
* [`disallowed_method`] [#6081](https://github.com/rust-lang/rust-clippy/pull/6081)
* [`inline_asm_x86_att_syntax`] [#6092](https://github.com/rust-lang/rust-clippy/pull/6092)
* [`inline_asm_x86_intel_syntax`] [#6092](https://github.com/rust-lang/rust-clippy/pull/6092)
* [`from_iter_instead_of_collect`] [#6101](https://github.com/rust-lang/rust-clippy/pull/6101)
* [`mut_mutex_lock`] [#6103](https://github.com/rust-lang/rust-clippy/pull/6103)
* [`single_element_loop`] [#6109](https://github.com/rust-lang/rust-clippy/pull/6109)
* [`manual_unwrap_or`] [#6123](https://github.com/rust-lang/rust-clippy/pull/6123)
* [`large_types_passed_by_value`] [#6135](https://github.com/rust-lang/rust-clippy/pull/6135)
* [`result_unit_err`] [#6157](https://github.com/rust-lang/rust-clippy/pull/6157)
* [`ref_option_ref`] [#6165](https://github.com/rust-lang/rust-clippy/pull/6165)
* [`manual_range_contains`] [#6177](https://github.com/rust-lang/rust-clippy/pull/6177)
* [`unusual_byte_groupings`] [#6183](https://github.com/rust-lang/rust-clippy/pull/6183)
* [`comparison_to_empty`] [#6226](https://github.com/rust-lang/rust-clippy/pull/6226)
* [`map_collect_result_unit`] [#6227](https://github.com/rust-lang/rust-clippy/pull/6227)
* [`manual_ok_or`] [#6233](https://github.com/rust-lang/rust-clippy/pull/6233)

### Moves and Deprecations

* Rename `single_char_push_str` to [`single_char_add_str`]
  [#6037](https://github.com/rust-lang/rust-clippy/pull/6037)
* Rename `zero_width_space` to [`invisible_characters`]
  [#6105](https://github.com/rust-lang/rust-clippy/pull/6105)
* Deprecate [`drop_bounds`] (uplifted)
  [#6111](https://github.com/rust-lang/rust-clippy/pull/6111)
* Move [`string_lit_as_bytes`] to `nursery`
  [#6117](https://github.com/rust-lang/rust-clippy/pull/6117)
* Move [`rc_buffer`] to `restriction`
  [#6128](https://github.com/rust-lang/rust-clippy/pull/6128)

### Enhancements

* [`manual_memcpy`]: Also lint when there are loop counters (and produce a
  reliable suggestion)
  [#5727](https://github.com/rust-lang/rust-clippy/pull/5727)
* [`single_char_add_str`]: Also lint on `String::insert_str`
  [#6037](https://github.com/rust-lang/rust-clippy/pull/6037)
* [`invisible_characters`]: Also lint the characters `\u{AD}` and `\u{2060}`
  [#6105](https://github.com/rust-lang/rust-clippy/pull/6105)
* [`eq_op`]: Also lint on the `assert_*!` macro family
  [#6167](https://github.com/rust-lang/rust-clippy/pull/6167)
* [`items_after_statements`]: Also lint in local macro expansions
  [#6176](https://github.com/rust-lang/rust-clippy/pull/6176)
* [`unnecessary_cast`]: Also lint casts on integer and float literals
  [#6187](https://github.com/rust-lang/rust-clippy/pull/6187)
* [`manual_unwrap_or`]: Also lint `Result::unwrap_or`
  [#6190](https://github.com/rust-lang/rust-clippy/pull/6190)
* [`match_like_matches_macro`]: Also lint when `match` has more than two arms
  [#6216](https://github.com/rust-lang/rust-clippy/pull/6216)
* [`integer_arithmetic`]: Better handle `/` an `%` operators
  [#6229](https://github.com/rust-lang/rust-clippy/pull/6229)

### False Positive Fixes

* [`needless_lifetimes`]: Bail out if the function has a `where` clause with the
  lifetime [#5978](https://github.com/rust-lang/rust-clippy/pull/5978)
* [`explicit_counter_loop`]: No longer lints, when loop counter is used after it
  is incremented [#6076](https://github.com/rust-lang/rust-clippy/pull/6076)
* [`or_fun_call`]: Revert changes addressing the handling of `const fn`
  [#6077](https://github.com/rust-lang/rust-clippy/pull/6077)
* [`needless_range_loop`]: No longer lints, when the iterable is used in the
  range [#6102](https://github.com/rust-lang/rust-clippy/pull/6102)
* [`inconsistent_digit_grouping`]: Fix bug when using floating point exponent
  [#6104](https://github.com/rust-lang/rust-clippy/pull/6104)
* [`mistyped_literal_suffixes`]: No longer lints on the fractional part of a
  float (e.g. `713.32_64`)
  [#6114](https://github.com/rust-lang/rust-clippy/pull/6114)
* [`invalid_regex`]: No longer lint on unicode characters within `bytes::Regex`
  [#6132](https://github.com/rust-lang/rust-clippy/pull/6132)
* [`boxed_local`]: No longer lints on `extern fn` arguments
  [#6133](https://github.com/rust-lang/rust-clippy/pull/6133)
* [`needless_lifetimes`]: Fix regression, where lifetime is used in `where`
  clause [#6198](https://github.com/rust-lang/rust-clippy/pull/6198)

### Suggestion Fixes/Improvements

* [`unnecessary_sort_by`]: Avoid dereferencing the suggested closure parameter
  [#6078](https://github.com/rust-lang/rust-clippy/pull/6078)
* [`needless_arbitrary_self_type`]: Correctly handle expanded code
  [#6093](https://github.com/rust-lang/rust-clippy/pull/6093)
* [`useless_format`]: Preserve raw strings in suggestion
  [#6151](https://github.com/rust-lang/rust-clippy/pull/6151)
* [`empty_loop`]: Suggest alternatives
  [#6162](https://github.com/rust-lang/rust-clippy/pull/6162)
* [`borrowed_box`]: Correctly add parentheses in suggestion
  [#6200](https://github.com/rust-lang/rust-clippy/pull/6200)
* [`unused_unit`]: Improve suggestion formatting
  [#6247](https://github.com/rust-lang/rust-clippy/pull/6247)

### Documentation Improvements

* Some doc improvements:
    * [`rc_buffer`] [#6090](https://github.com/rust-lang/rust-clippy/pull/6090)
    * [`empty_loop`] [#6162](https://github.com/rust-lang/rust-clippy/pull/6162)
* [`doc_markdown`]: Document problematic link text style
  [#6107](https://github.com/rust-lang/rust-clippy/pull/6107)

## Rust 1.48

Released 2020-11-19

[09bd400...e636b88](https://github.com/rust-lang/rust-clippy/compare/09bd400...e636b88)

### New lints

* [`self_assignment`] [#5894](https://github.com/rust-lang/rust-clippy/pull/5894)
* [`unnecessary_lazy_evaluations`] [#5720](https://github.com/rust-lang/rust-clippy/pull/5720)
* [`manual_strip`] [#6038](https://github.com/rust-lang/rust-clippy/pull/6038)
* [`map_err_ignore`] [#5998](https://github.com/rust-lang/rust-clippy/pull/5998)
* [`rc_buffer`] [#6044](https://github.com/rust-lang/rust-clippy/pull/6044)
* [`to_string_in_display`] [#5831](https://github.com/rust-lang/rust-clippy/pull/5831)
* `single_char_push_str` [#5881](https://github.com/rust-lang/rust-clippy/pull/5881)

### Moves and Deprecations

* Downgrade [`verbose_bit_mask`] to pedantic
  [#6036](https://github.com/rust-lang/rust-clippy/pull/6036)

### Enhancements

* Extend [`precedence`] to handle chains of methods combined with unary negation
  [#5928](https://github.com/rust-lang/rust-clippy/pull/5928)
* [`useless_vec`]: add a configuration value for the maximum allowed size on the stack
  [#5907](https://github.com/rust-lang/rust-clippy/pull/5907)
* [`suspicious_arithmetic_impl`]: extend to implementations of `BitAnd`, `BitOr`, `BitXor`, `Rem`, `Shl`, and `Shr`
  [#5884](https://github.com/rust-lang/rust-clippy/pull/5884)
* [`invalid_atomic_ordering`]: detect misuse of `compare_exchange`, `compare_exchange_weak`, and `fetch_update`
  [#6025](https://github.com/rust-lang/rust-clippy/pull/6025)
* Avoid [`redundant_pattern_matching`] triggering in macros
  [#6069](https://github.com/rust-lang/rust-clippy/pull/6069)
* [`option_if_let_else`]: distinguish pure from impure `else` expressions
  [#5937](https://github.com/rust-lang/rust-clippy/pull/5937)
* [`needless_doctest_main`]: parse doctests instead of using textual search
  [#5912](https://github.com/rust-lang/rust-clippy/pull/5912)
* [`wildcard_imports`]: allow `prelude` to appear in any segment of an import
  [#5929](https://github.com/rust-lang/rust-clippy/pull/5929)
* Re-enable [`len_zero`] for ranges now that `range_is_empty` is stable
  [#5961](https://github.com/rust-lang/rust-clippy/pull/5961)
* [`option_as_ref_deref`]: catch fully-qualified calls to `Deref::deref` and `DerefMut::deref_mut`
  [#5933](https://github.com/rust-lang/rust-clippy/pull/5933)

### False Positive Fixes

* [`useless_attribute`]: permit allowing [`wildcard_imports`] and [`enum_glob_use`]
  [#5994](https://github.com/rust-lang/rust-clippy/pull/5994)
* [`transmute_ptr_to_ptr`]: avoid suggesting dereferencing raw pointers in const contexts
  [#5999](https://github.com/rust-lang/rust-clippy/pull/5999)
* [`redundant_closure_call`]: take into account usages of the closure in nested functions and closures
  [#5920](https://github.com/rust-lang/rust-clippy/pull/5920)
* Fix false positive in [`borrow_interior_mutable_const`] when referencing a field behind a pointer
  [#5949](https://github.com/rust-lang/rust-clippy/pull/5949)
* [`doc_markdown`]: allow using "GraphQL" without backticks
  [#5996](https://github.com/rust-lang/rust-clippy/pull/5996)
* [`to_string_in_display`]: avoid linting when calling `to_string()` on anything that is not `self`
  [#5971](https://github.com/rust-lang/rust-clippy/pull/5971)
* [`indexing_slicing`] and [`out_of_bounds_indexing`] treat references to arrays as arrays
  [#6034](https://github.com/rust-lang/rust-clippy/pull/6034)
* [`should_implement_trait`]: ignore methods with lifetime parameters
  [#5725](https://github.com/rust-lang/rust-clippy/pull/5725)
* [`needless_return`]: avoid linting if a temporary borrows a local variable
  [#5903](https://github.com/rust-lang/rust-clippy/pull/5903)
* Restrict [`unnecessary_sort_by`] to non-reference, Copy types
  [#6006](https://github.com/rust-lang/rust-clippy/pull/6006)
* Avoid suggesting `from_bits`/`to_bits` in const contexts in [`transmute_int_to_float`]
  [#5919](https://github.com/rust-lang/rust-clippy/pull/5919)
* [`declare_interior_mutable_const`] and [`borrow_interior_mutable_const`]: improve detection of interior mutable types
  [#6046](https://github.com/rust-lang/rust-clippy/pull/6046)

### Suggestion Fixes/Improvements

* [`let_and_return`]: add a cast to the suggestion when the return expression has adjustments
  [#5946](https://github.com/rust-lang/rust-clippy/pull/5946)
* [`useless_conversion`]: show the type in the error message
  [#6035](https://github.com/rust-lang/rust-clippy/pull/6035)
* [`unnecessary_mut_passed`]: discriminate between functions and methods in the error message
  [#5892](https://github.com/rust-lang/rust-clippy/pull/5892)
* [`float_cmp`] and [`float_cmp_const`]: change wording to make margin of error less ambiguous
  [#6043](https://github.com/rust-lang/rust-clippy/pull/6043)
* [`default_trait_access`]: do not use unnecessary type parameters in the suggestion
  [#5993](https://github.com/rust-lang/rust-clippy/pull/5993)
* [`collapsible_if`]: don't use expanded code in the suggestion
  [#5992](https://github.com/rust-lang/rust-clippy/pull/5992)
* Do not suggest empty format strings in [`print_with_newline`] and [`write_with_newline`]
  [#6042](https://github.com/rust-lang/rust-clippy/pull/6042)
* [`unit_arg`]: improve the readability of the suggestion
  [#5931](https://github.com/rust-lang/rust-clippy/pull/5931)
* [`stable_sort_primitive`]: print the type that is being sorted in the lint message
  [#5935](https://github.com/rust-lang/rust-clippy/pull/5935)
* Show line count and max lines in [`too_many_lines`] lint message
  [#6009](https://github.com/rust-lang/rust-clippy/pull/6009)
* Keep parentheses in the suggestion of [`useless_conversion`] where applicable
  [#5900](https://github.com/rust-lang/rust-clippy/pull/5900)
* [`option_map_unit_fn`] and [`result_map_unit_fn`]: print the unit type `()` explicitly
  [#6024](https://github.com/rust-lang/rust-clippy/pull/6024)
* [`redundant_allocation`]: suggest replacing `Rc<Box<T>>` with `Rc<T>`
  [#5899](https://github.com/rust-lang/rust-clippy/pull/5899)
* Make lint messages adhere to rustc dev guide conventions
  [#5893](https://github.com/rust-lang/rust-clippy/pull/5893)

### ICE Fixes

* Fix ICE in [`repeat_once`]
  [#5948](https://github.com/rust-lang/rust-clippy/pull/5948)

### Documentation Improvements

* [`mutable_key_type`]: explain potential for false positives when the interior mutable type is not accessed in the `Hash` implementation
  [#6019](https://github.com/rust-lang/rust-clippy/pull/6019)
* [`unnecessary_mut_passed`]: fix typo
  [#5913](https://github.com/rust-lang/rust-clippy/pull/5913)
* Add example of false positive to [`ptr_arg`] docs.
  [#5885](https://github.com/rust-lang/rust-clippy/pull/5885)
* [`box_vec`], [`vec_box`] and [`borrowed_box`]: add link to the documentation of `Box`
  [#6023](https://github.com/rust-lang/rust-clippy/pull/6023)

## Rust 1.47

Released 2020-10-08

[c2c07fa...09bd400](https://github.com/rust-lang/rust-clippy/compare/c2c07fa...09bd400)

### New lints

* [`derive_ord_xor_partial_ord`] [#5848](https://github.com/rust-lang/rust-clippy/pull/5848)
* [`trait_duplication_in_bounds`] [#5852](https://github.com/rust-lang/rust-clippy/pull/5852)
* [`map_identity`] [#5694](https://github.com/rust-lang/rust-clippy/pull/5694)
* [`unit_return_expecting_ord`] [#5737](https://github.com/rust-lang/rust-clippy/pull/5737)
* [`pattern_type_mismatch`] [#4841](https://github.com/rust-lang/rust-clippy/pull/4841)
* [`repeat_once`] [#5773](https://github.com/rust-lang/rust-clippy/pull/5773)
* [`same_item_push`] [#5825](https://github.com/rust-lang/rust-clippy/pull/5825)
* [`needless_arbitrary_self_type`] [#5869](https://github.com/rust-lang/rust-clippy/pull/5869)
* [`match_like_matches_macro`] [#5769](https://github.com/rust-lang/rust-clippy/pull/5769)
* [`stable_sort_primitive`] [#5809](https://github.com/rust-lang/rust-clippy/pull/5809)
* [`blanket_clippy_restriction_lints`] [#5750](https://github.com/rust-lang/rust-clippy/pull/5750)
* [`option_if_let_else`] [#5301](https://github.com/rust-lang/rust-clippy/pull/5301)

### Moves and Deprecations

* Deprecate [`regex_macro`] lint
  [#5760](https://github.com/rust-lang/rust-clippy/pull/5760)
* Move [`range_minus_one`] to `pedantic`
  [#5752](https://github.com/rust-lang/rust-clippy/pull/5752)

### Enhancements

* Improve [`needless_collect`] by catching `collect` calls followed by `iter` or `into_iter` calls
  [#5837](https://github.com/rust-lang/rust-clippy/pull/5837)
* [`panic`], [`todo`], [`unimplemented`] and [`unreachable`] now detect calls with formatting
  [#5811](https://github.com/rust-lang/rust-clippy/pull/5811)
* Detect more cases of [`suboptimal_flops`] and [`imprecise_flops`]
  [#5443](https://github.com/rust-lang/rust-clippy/pull/5443)
* Handle asymmetrical implementations of `PartialEq` in [`cmp_owned`]
  [#5701](https://github.com/rust-lang/rust-clippy/pull/5701)
* Make it possible to allow [`unsafe_derive_deserialize`]
  [#5870](https://github.com/rust-lang/rust-clippy/pull/5870)
* Catch `ord.min(a).max(b)` where a < b in [`min_max`]
  [#5871](https://github.com/rust-lang/rust-clippy/pull/5871)
* Make [`clone_on_copy`] suggestion machine applicable
  [#5745](https://github.com/rust-lang/rust-clippy/pull/5745)
* Enable [`len_zero`] on ranges now that `is_empty` is stable on them
  [#5961](https://github.com/rust-lang/rust-clippy/pull/5961)

### False Positive Fixes

* Avoid triggering [`or_fun_call`] with const fns that take no arguments
  [#5889](https://github.com/rust-lang/rust-clippy/pull/5889)
* Fix [`redundant_closure_call`] false positive for closures that have multiple calls
  [#5800](https://github.com/rust-lang/rust-clippy/pull/5800)
* Don't lint cases involving `ManuallyDrop` in [`redundant_clone`]
  [#5824](https://github.com/rust-lang/rust-clippy/pull/5824)
* Treat a single expression the same as a single statement in the 2nd arm of a match in [`single_match_else`]
  [#5771](https://github.com/rust-lang/rust-clippy/pull/5771)
* Don't trigger [`unnested_or_patterns`] if the feature `or_patterns` is not enabled
  [#5758](https://github.com/rust-lang/rust-clippy/pull/5758)
* Avoid linting if key borrows in [`unnecessary_sort_by`]
  [#5756](https://github.com/rust-lang/rust-clippy/pull/5756)
* Consider `Try` impl for `Poll` when generating suggestions in [`try_err`]
  [#5857](https://github.com/rust-lang/rust-clippy/pull/5857)
* Take input lifetimes into account in `manual_async_fn`
  [#5859](https://github.com/rust-lang/rust-clippy/pull/5859)
* Fix multiple false positives in [`type_repetition_in_bounds`] and add a configuration option
  [#5761](https://github.com/rust-lang/rust-clippy/pull/5761)
* Limit the [`suspicious_arithmetic_impl`] lint to one binary operation
  [#5820](https://github.com/rust-lang/rust-clippy/pull/5820)

### Suggestion Fixes/Improvements

* Improve readability of [`shadow_unrelated`] suggestion by truncating the RHS snippet
  [#5788](https://github.com/rust-lang/rust-clippy/pull/5788)
* Suggest `filter_map` instead of `flat_map` when mapping to `Option` in [`map_flatten`]
  [#5846](https://github.com/rust-lang/rust-clippy/pull/5846)
* Ensure suggestion is shown correctly for long method call chains in [`iter_nth_zero`]
  [#5793](https://github.com/rust-lang/rust-clippy/pull/5793)
* Drop borrow operator in suggestions of [`redundant_pattern_matching`]
  [#5815](https://github.com/rust-lang/rust-clippy/pull/5815)
* Add suggestion for [`iter_skip_next`]
  [#5843](https://github.com/rust-lang/rust-clippy/pull/5843)
* Improve [`collapsible_if`] fix suggestion
  [#5732](https://github.com/rust-lang/rust-clippy/pull/5732)

### ICE Fixes

* Fix ICE caused by [`needless_collect`]
  [#5877](https://github.com/rust-lang/rust-clippy/pull/5877)
* Fix ICE caused by [`unnested_or_patterns`]
  [#5784](https://github.com/rust-lang/rust-clippy/pull/5784)

### Documentation Improvements

* Fix grammar of [`await_holding_lock`] documentation
  [#5748](https://github.com/rust-lang/rust-clippy/pull/5748)

### Others

* Make lints adhere to the rustc dev guide
  [#5888](https://github.com/rust-lang/rust-clippy/pull/5888)

## Rust 1.46

Released 2020-08-27

[7ea7cd1...c2c07fa](https://github.com/rust-lang/rust-clippy/compare/7ea7cd1...c2c07fa)

### New lints

* [`unnested_or_patterns`] [#5378](https://github.com/rust-lang/rust-clippy/pull/5378)
* [`iter_next_slice`] [#5597](https://github.com/rust-lang/rust-clippy/pull/5597)
* [`unnecessary_sort_by`] [#5623](https://github.com/rust-lang/rust-clippy/pull/5623)
* [`vec_resize_to_zero`] [#5637](https://github.com/rust-lang/rust-clippy/pull/5637)

### Moves and Deprecations

* Move [`cast_ptr_alignment`] to pedantic [#5667](https://github.com/rust-lang/rust-clippy/pull/5667)

### Enhancements

* Improve [`mem_replace_with_uninit`] lint [#5695](https://github.com/rust-lang/rust-clippy/pull/5695)

### False Positive Fixes

* [`len_zero`]: Avoid linting ranges when the `range_is_empty` feature is not enabled
  [#5656](https://github.com/rust-lang/rust-clippy/pull/5656)
* [`let_and_return`]: Don't lint if a temporary borrow is involved
  [#5680](https://github.com/rust-lang/rust-clippy/pull/5680)
* [`reversed_empty_ranges`]: Avoid linting `N..N` in for loop arguments in
  [#5692](https://github.com/rust-lang/rust-clippy/pull/5692)
* [`if_same_then_else`]: Don't assume multiplication is always commutative
  [#5702](https://github.com/rust-lang/rust-clippy/pull/5702)
* [`blacklisted_name`]: Remove `bar` from the default configuration
  [#5712](https://github.com/rust-lang/rust-clippy/pull/5712)
* [`redundant_pattern_matching`]: Avoid suggesting non-`const fn` calls in const contexts
  [#5724](https://github.com/rust-lang/rust-clippy/pull/5724)

### Suggestion Fixes/Improvements

* Fix suggestion of [`unit_arg`] lint, so that it suggest semantic equivalent code
  [#4455](https://github.com/rust-lang/rust-clippy/pull/4455)
* Add auto applicable suggestion to [`macro_use_imports`]
  [#5279](https://github.com/rust-lang/rust-clippy/pull/5279)

### ICE Fixes

* Fix ICE in the `consts` module of Clippy [#5709](https://github.com/rust-lang/rust-clippy/pull/5709)

### Documentation Improvements

* Improve code examples across multiple lints [#5664](https://github.com/rust-lang/rust-clippy/pull/5664)

### Others

* Introduce a `--rustc` flag to `clippy-driver`, which turns `clippy-driver`
  into `rustc` and passes all the given arguments to `rustc`. This is especially
  useful for tools that need the `rustc` version Clippy was compiled with,
  instead of the Clippy version. E.g. `clippy-driver --rustc --version` will
  print the output of `rustc --version`.
  [#5178](https://github.com/rust-lang/rust-clippy/pull/5178)
* New issue templates now make it easier to complain if Clippy is too annoying
  or not annoying enough! [#5735](https://github.com/rust-lang/rust-clippy/pull/5735)

## Rust 1.45

Released 2020-07-16

[891e1a8...7ea7cd1](https://github.com/rust-lang/rust-clippy/compare/891e1a8...7ea7cd1)

### New lints

* [`match_wildcard_for_single_variants`] [#5582](https://github.com/rust-lang/rust-clippy/pull/5582)
* [`unsafe_derive_deserialize`] [#5493](https://github.com/rust-lang/rust-clippy/pull/5493)
* [`if_let_mutex`] [#5332](https://github.com/rust-lang/rust-clippy/pull/5332)
* [`mismatched_target_os`] [#5506](https://github.com/rust-lang/rust-clippy/pull/5506)
* [`await_holding_lock`] [#5439](https://github.com/rust-lang/rust-clippy/pull/5439)
* [`match_on_vec_items`] [#5522](https://github.com/rust-lang/rust-clippy/pull/5522)
* [`manual_async_fn`] [#5576](https://github.com/rust-lang/rust-clippy/pull/5576)
* [`reversed_empty_ranges`] [#5583](https://github.com/rust-lang/rust-clippy/pull/5583)
* [`manual_non_exhaustive`] [#5550](https://github.com/rust-lang/rust-clippy/pull/5550)

### Moves and Deprecations

* Downgrade [`match_bool`] to pedantic [#5408](https://github.com/rust-lang/rust-clippy/pull/5408)
* Downgrade [`match_wild_err_arm`] to pedantic and update help messages. [#5622](https://github.com/rust-lang/rust-clippy/pull/5622)
* Downgrade [`useless_let_if_seq`] to nursery. [#5599](https://github.com/rust-lang/rust-clippy/pull/5599)
* Generalize `option_and_then_some` and rename to [`bind_instead_of_map`]. [#5529](https://github.com/rust-lang/rust-clippy/pull/5529)
* Rename `identity_conversion` to [`useless_conversion`]. [#5568](https://github.com/rust-lang/rust-clippy/pull/5568)
* Merge `block_in_if_condition_expr` and `block_in_if_condition_stmt` into [`blocks_in_if_conditions`].
[#5563](https://github.com/rust-lang/rust-clippy/pull/5563)
* Merge `option_map_unwrap_or`, `option_map_unwrap_or_else` and `result_map_unwrap_or_else` into [`map_unwrap_or`].
[#5563](https://github.com/rust-lang/rust-clippy/pull/5563)
* Merge `option_unwrap_used` and `result_unwrap_used` into [`unwrap_used`].
[#5563](https://github.com/rust-lang/rust-clippy/pull/5563)
* Merge `option_expect_used` and `result_expect_used` into [`expect_used`].
[#5563](https://github.com/rust-lang/rust-clippy/pull/5563)
* Merge `for_loop_over_option` and `for_loop_over_result` into [`for_loops_over_fallibles`].
[#5563](https://github.com/rust-lang/rust-clippy/pull/5563)

### Enhancements

* Avoid running cargo lints when not enabled to improve performance. [#5505](https://github.com/rust-lang/rust-clippy/pull/5505)
* Extend [`useless_conversion`] with `TryFrom` and `TryInto`. [#5631](https://github.com/rust-lang/rust-clippy/pull/5631)
* Lint also in type parameters and where clauses in [`unused_unit`]. [#5592](https://github.com/rust-lang/rust-clippy/pull/5592)
* Do not suggest deriving `Default` in [`new_without_default`]. [#5616](https://github.com/rust-lang/rust-clippy/pull/5616)

### False Positive Fixes

* [`while_let_on_iterator`] [#5525](https://github.com/rust-lang/rust-clippy/pull/5525)
* [`empty_line_after_outer_attr`] [#5609](https://github.com/rust-lang/rust-clippy/pull/5609)
* [`unnecessary_unwrap`] [#5558](https://github.com/rust-lang/rust-clippy/pull/5558)
* [`comparison_chain`] [#5596](https://github.com/rust-lang/rust-clippy/pull/5596)
* Don't trigger [`used_underscore_binding`] in await desugaring. [#5535](https://github.com/rust-lang/rust-clippy/pull/5535)
* Don't trigger [`borrowed_box`] on mutable references. [#5491](https://github.com/rust-lang/rust-clippy/pull/5491)
* Allow `1 << 0` in [`identity_op`]. [#5602](https://github.com/rust-lang/rust-clippy/pull/5602)
* Allow `use super::*;` glob imports in [`wildcard_imports`]. [#5564](https://github.com/rust-lang/rust-clippy/pull/5564)
* Whitelist more words in [`doc_markdown`]. [#5611](https://github.com/rust-lang/rust-clippy/pull/5611)
* Skip dev and build deps in [`multiple_crate_versions`]. [#5636](https://github.com/rust-lang/rust-clippy/pull/5636)
* Honor `allow` attribute on arguments in [`ptr_arg`]. [#5647](https://github.com/rust-lang/rust-clippy/pull/5647)
* Honor lint level attributes for [`redundant_field_names`], [`just_underscores_and_digits`], [`many_single_char_names`]
and [`similar_names`]. [#5651](https://github.com/rust-lang/rust-clippy/pull/5651)
* Ignore calls to `len` in [`or_fun_call`]. [#4429](https://github.com/rust-lang/rust-clippy/pull/4429)

### Suggestion Improvements

* Simplify suggestions in [`manual_memcpy`]. [#5536](https://github.com/rust-lang/rust-clippy/pull/5536)
* Fix suggestion in [`redundant_pattern_matching`] for macros. [#5511](https://github.com/rust-lang/rust-clippy/pull/5511)
* Avoid suggesting `copied()` for mutable references in [`map_clone`]. [#5530](https://github.com/rust-lang/rust-clippy/pull/5530)
* Improve help message for [`clone_double_ref`]. [#5547](https://github.com/rust-lang/rust-clippy/pull/5547)

### ICE Fixes

* Fix ICE caused in unwrap module. [#5590](https://github.com/rust-lang/rust-clippy/pull/5590)
* Fix ICE on rustc test issue-69020-assoc-const-arith-overflow.rs [#5499](https://github.com/rust-lang/rust-clippy/pull/5499)

### Documentation

* Clarify the documentation of [`unnecessary_mut_passed`]. [#5639](https://github.com/rust-lang/rust-clippy/pull/5639)
* Extend example for [`unneeded_field_pattern`]. [#5541](https://github.com/rust-lang/rust-clippy/pull/5541)

## Rust 1.44

Released 2020-06-04

[204bb9b...891e1a8](https://github.com/rust-lang/rust-clippy/compare/204bb9b...891e1a8)

### New lints

* [`explicit_deref_methods`] [#5226](https://github.com/rust-lang/rust-clippy/pull/5226)
* [`implicit_saturating_sub`] [#5427](https://github.com/rust-lang/rust-clippy/pull/5427)
* [`macro_use_imports`] [#5230](https://github.com/rust-lang/rust-clippy/pull/5230)
* [`verbose_file_reads`] [#5272](https://github.com/rust-lang/rust-clippy/pull/5272)
* [`future_not_send`] [#5423](https://github.com/rust-lang/rust-clippy/pull/5423)
* [`redundant_pub_crate`] [#5319](https://github.com/rust-lang/rust-clippy/pull/5319)
* [`large_const_arrays`] [#5248](https://github.com/rust-lang/rust-clippy/pull/5248)
* [`result_map_or_into_option`] [#5415](https://github.com/rust-lang/rust-clippy/pull/5415)
* [`redundant_allocation`] [#5349](https://github.com/rust-lang/rust-clippy/pull/5349)
* [`fn_address_comparisons`] [#5294](https://github.com/rust-lang/rust-clippy/pull/5294)
* [`vtable_address_comparisons`] [#5294](https://github.com/rust-lang/rust-clippy/pull/5294)


### Moves and Deprecations

* Deprecate [`replace_consts`] lint [#5380](https://github.com/rust-lang/rust-clippy/pull/5380)
* Move [`cognitive_complexity`] to nursery [#5428](https://github.com/rust-lang/rust-clippy/pull/5428)
* Move [`useless_transmute`] to nursery [#5364](https://github.com/rust-lang/rust-clippy/pull/5364)
* Downgrade [`inefficient_to_string`] to pedantic [#5412](https://github.com/rust-lang/rust-clippy/pull/5412)
* Downgrade [`option_option`] to pedantic [#5401](https://github.com/rust-lang/rust-clippy/pull/5401)
* Downgrade [`unreadable_literal`] to pedantic [#5419](https://github.com/rust-lang/rust-clippy/pull/5419)
* Downgrade [`let_unit_value`] to pedantic [#5409](https://github.com/rust-lang/rust-clippy/pull/5409)
* Downgrade [`trivially_copy_pass_by_ref`] to pedantic [#5410](https://github.com/rust-lang/rust-clippy/pull/5410)
* Downgrade [`implicit_hasher`] to pedantic [#5411](https://github.com/rust-lang/rust-clippy/pull/5411)

### Enhancements

* On _nightly_ you can now use `cargo clippy --fix -Z unstable-options` to
  auto-fix lints that support this [#5363](https://github.com/rust-lang/rust-clippy/pull/5363)
* Make [`redundant_clone`] also trigger on cases where the cloned value is not
  consumed. [#5304](https://github.com/rust-lang/rust-clippy/pull/5304)
* Expand [`integer_arithmetic`] to also disallow bit-shifting [#5430](https://github.com/rust-lang/rust-clippy/pull/5430)
* [`option_as_ref_deref`] now detects more deref cases [#5425](https://github.com/rust-lang/rust-clippy/pull/5425)
* [`large_enum_variant`] now report the sizes of the largest and second-largest variants [#5466](https://github.com/rust-lang/rust-clippy/pull/5466)
* [`bool_comparison`] now also checks for inequality comparisons that can be
  written more concisely [#5365](https://github.com/rust-lang/rust-clippy/pull/5365)
* Expand [`clone_on_copy`] to work in method call arguments as well [#5441](https://github.com/rust-lang/rust-clippy/pull/5441)
* [`redundant_pattern_matching`] now also handles `while let` [#5483](https://github.com/rust-lang/rust-clippy/pull/5483)
* [`integer_arithmetic`] now also lints references of integers [#5329](https://github.com/rust-lang/rust-clippy/pull/5329)
* Expand [`float_cmp_const`] to also work on arrays [#5345](https://github.com/rust-lang/rust-clippy/pull/5345)
* Trigger [`map_flatten`] when map is called on an `Option` [#5473](https://github.com/rust-lang/rust-clippy/pull/5473)

### False Positive Fixes

* [`many_single_char_names`] [#5468](https://github.com/rust-lang/rust-clippy/pull/5468)
* [`should_implement_trait`] [#5437](https://github.com/rust-lang/rust-clippy/pull/5437)
* [`unused_self`] [#5387](https://github.com/rust-lang/rust-clippy/pull/5387)
* [`redundant_clone`] [#5453](https://github.com/rust-lang/rust-clippy/pull/5453)
* [`precedence`] [#5445](https://github.com/rust-lang/rust-clippy/pull/5445)
* [`suspicious_op_assign_impl`] [#5424](https://github.com/rust-lang/rust-clippy/pull/5424)
* [`needless_lifetimes`] [#5293](https://github.com/rust-lang/rust-clippy/pull/5293)
* [`redundant_pattern`] [#5287](https://github.com/rust-lang/rust-clippy/pull/5287)
* [`inconsistent_digit_grouping`] [#5451](https://github.com/rust-lang/rust-clippy/pull/5451)


### Suggestion Improvements

* Improved [`question_mark`] lint suggestion so that it doesn't add redundant `as_ref()` [#5481](https://github.com/rust-lang/rust-clippy/pull/5481)
* Improve the suggested placeholder in [`option_map_unit_fn`] [#5292](https://github.com/rust-lang/rust-clippy/pull/5292)
* Improve suggestion for [`match_single_binding`] when triggered inside a closure [#5350](https://github.com/rust-lang/rust-clippy/pull/5350)

### ICE Fixes

* Handle the unstable `trivial_bounds` feature [#5296](https://github.com/rust-lang/rust-clippy/pull/5296)
* `shadow_*` lints [#5297](https://github.com/rust-lang/rust-clippy/pull/5297)

### Documentation

* Fix documentation generation for configurable lints [#5353](https://github.com/rust-lang/rust-clippy/pull/5353)
* Update documentation for [`new_ret_no_self`] [#5448](https://github.com/rust-lang/rust-clippy/pull/5448)
* The documentation for [`option_option`] now suggest using a tri-state enum [#5403](https://github.com/rust-lang/rust-clippy/pull/5403)
* Fix bit mask example in [`verbose_bit_mask`] documentation [#5454](https://github.com/rust-lang/rust-clippy/pull/5454)
* [`wildcard_imports`] documentation now mentions that `use ...::prelude::*` is
  not linted [#5312](https://github.com/rust-lang/rust-clippy/pull/5312)

## Rust 1.43

Released 2020-04-23

[4ee1206...204bb9b](https://github.com/rust-lang/rust-clippy/compare/4ee1206...204bb9b)

### New lints

* [`imprecise_flops`] [#4897](https://github.com/rust-lang/rust-clippy/pull/4897)
* [`suboptimal_flops`] [#4897](https://github.com/rust-lang/rust-clippy/pull/4897)
* [`wildcard_imports`] [#5029](https://github.com/rust-lang/rust-clippy/pull/5029)
* [`single_component_path_imports`] [#5058](https://github.com/rust-lang/rust-clippy/pull/5058)
* [`match_single_binding`] [#5061](https://github.com/rust-lang/rust-clippy/pull/5061)
* [`let_underscore_lock`] [#5101](https://github.com/rust-lang/rust-clippy/pull/5101)
* [`struct_excessive_bools`] [#5125](https://github.com/rust-lang/rust-clippy/pull/5125)
* [`fn_params_excessive_bools`] [#5125](https://github.com/rust-lang/rust-clippy/pull/5125)
* [`option_env_unwrap`] [#5148](https://github.com/rust-lang/rust-clippy/pull/5148)
* [`lossy_float_literal`] [#5202](https://github.com/rust-lang/rust-clippy/pull/5202)
* [`rest_pat_in_fully_bound_structs`] [#5258](https://github.com/rust-lang/rust-clippy/pull/5258)

### Moves and Deprecations

* Move [`unneeded_field_pattern`] to pedantic group [#5200](https://github.com/rust-lang/rust-clippy/pull/5200)

### Enhancements

* Make [`missing_errors_doc`] lint also trigger on `async` functions
  [#5181](https://github.com/rust-lang/rust-clippy/pull/5181)
* Add more constants to [`approx_constant`] [#5193](https://github.com/rust-lang/rust-clippy/pull/5193)
* Extend [`question_mark`] lint [#5266](https://github.com/rust-lang/rust-clippy/pull/5266)

### False Positive Fixes

* [`use_debug`] [#5047](https://github.com/rust-lang/rust-clippy/pull/5047)
* [`unnecessary_unwrap`] [#5132](https://github.com/rust-lang/rust-clippy/pull/5132)
* [`zero_prefixed_literal`] [#5170](https://github.com/rust-lang/rust-clippy/pull/5170)
* [`missing_const_for_fn`] [#5216](https://github.com/rust-lang/rust-clippy/pull/5216)

### Suggestion Improvements

* Improve suggestion when blocks of code are suggested [#5134](https://github.com/rust-lang/rust-clippy/pull/5134)

### ICE Fixes

* `misc_early` lints [#5129](https://github.com/rust-lang/rust-clippy/pull/5129)
* [`missing_errors_doc`] [#5213](https://github.com/rust-lang/rust-clippy/pull/5213)
* Fix ICE when evaluating `usize`s [#5256](https://github.com/rust-lang/rust-clippy/pull/5256)

### Documentation

* Improve documentation of [`iter_nth_zero`]
* Add documentation pages for stable releases [#5171](https://github.com/rust-lang/rust-clippy/pull/5171)

### Others

* Clippy now completely runs on GitHub Actions [#5190](https://github.com/rust-lang/rust-clippy/pull/5190)


## Rust 1.42

Released 2020-03-12

[69f99e7...4ee1206](https://github.com/rust-lang/rust-clippy/compare/69f99e7...4ee1206)

### New lints

* [`filetype_is_file`] [#4543](https://github.com/rust-lang/rust-clippy/pull/4543)
* [`let_underscore_must_use`] [#4823](https://github.com/rust-lang/rust-clippy/pull/4823)
* [`modulo_arithmetic`] [#4867](https://github.com/rust-lang/rust-clippy/pull/4867)
* [`mem_replace_with_default`] [#4881](https://github.com/rust-lang/rust-clippy/pull/4881)
* [`mutable_key_type`] [#4885](https://github.com/rust-lang/rust-clippy/pull/4885)
* [`option_as_ref_deref`] [#4945](https://github.com/rust-lang/rust-clippy/pull/4945)
* [`wildcard_in_or_patterns`] [#4960](https://github.com/rust-lang/rust-clippy/pull/4960)
* [`iter_nth_zero`] [#4966](https://github.com/rust-lang/rust-clippy/pull/4966)
* [`invalid_atomic_ordering`] [#4999](https://github.com/rust-lang/rust-clippy/pull/4999)
* [`skip_while_next`] [#5067](https://github.com/rust-lang/rust-clippy/pull/5067)

### Moves and Deprecations

* Move [`transmute_float_to_int`] from nursery to complexity group
  [#5015](https://github.com/rust-lang/rust-clippy/pull/5015)
* Move [`range_plus_one`] to pedantic group [#5057](https://github.com/rust-lang/rust-clippy/pull/5057)
* Move [`debug_assert_with_mut_call`] to nursery group [#5106](https://github.com/rust-lang/rust-clippy/pull/5106)
* Deprecate [`unused_label`] [#4930](https://github.com/rust-lang/rust-clippy/pull/4930)

### Enhancements

* Lint vectored IO in [`unused_io_amount`] [#5027](https://github.com/rust-lang/rust-clippy/pull/5027)
* Make [`vec_box`] configurable by adding a size threshold [#5081](https://github.com/rust-lang/rust-clippy/pull/5081)
* Also lint constants in [`cmp_nan`] [#4910](https://github.com/rust-lang/rust-clippy/pull/4910)
* Fix false negative in [`expect_fun_call`] [#4915](https://github.com/rust-lang/rust-clippy/pull/4915)
* Fix false negative in [`redundant_clone`] [#5017](https://github.com/rust-lang/rust-clippy/pull/5017)

### False Positive Fixes

* [`map_clone`] [#4937](https://github.com/rust-lang/rust-clippy/pull/4937)
* [`replace_consts`] [#4977](https://github.com/rust-lang/rust-clippy/pull/4977)
* [`let_and_return`] [#5008](https://github.com/rust-lang/rust-clippy/pull/5008)
* [`eq_op`] [#5079](https://github.com/rust-lang/rust-clippy/pull/5079)
* [`possible_missing_comma`] [#5083](https://github.com/rust-lang/rust-clippy/pull/5083)
* [`debug_assert_with_mut_call`] [#5106](https://github.com/rust-lang/rust-clippy/pull/5106)
* Don't trigger [`let_underscore_must_use`] in external macros
  [#5082](https://github.com/rust-lang/rust-clippy/pull/5082)
* Don't trigger [`empty_loop`] in `no_std` crates [#5086](https://github.com/rust-lang/rust-clippy/pull/5086)

### Suggestion Improvements

* `option_map_unwrap_or` [#4634](https://github.com/rust-lang/rust-clippy/pull/4634)
* [`wildcard_enum_match_arm`] [#4934](https://github.com/rust-lang/rust-clippy/pull/4934)
* [`cognitive_complexity`] [#4935](https://github.com/rust-lang/rust-clippy/pull/4935)
* [`decimal_literal_representation`] [#4956](https://github.com/rust-lang/rust-clippy/pull/4956)
* [`unknown_clippy_lints`] [#4963](https://github.com/rust-lang/rust-clippy/pull/4963)
* [`explicit_into_iter_loop`] [#4978](https://github.com/rust-lang/rust-clippy/pull/4978)
* [`useless_attribute`] [#5022](https://github.com/rust-lang/rust-clippy/pull/5022)
* [`if_let_some_result`] [#5032](https://github.com/rust-lang/rust-clippy/pull/5032)

### ICE fixes

* [`unsound_collection_transmute`] [#4975](https://github.com/rust-lang/rust-clippy/pull/4975)

### Documentation

* Improve documentation of [`empty_enum`], [`replace_consts`], [`redundant_clone`], and [`iterator_step_by_zero`]


## Rust 1.41

Released 2020-01-30

[c8e3cfb...69f99e7](https://github.com/rust-lang/rust-clippy/compare/c8e3cfb...69f99e7)

* New Lints:
  * [`exit`] [#4697](https://github.com/rust-lang/rust-clippy/pull/4697)
  * [`to_digit_is_some`] [#4801](https://github.com/rust-lang/rust-clippy/pull/4801)
  * [`tabs_in_doc_comments`] [#4806](https://github.com/rust-lang/rust-clippy/pull/4806)
  * [`large_stack_arrays`] [#4807](https://github.com/rust-lang/rust-clippy/pull/4807)
  * [`same_functions_in_if_condition`] [#4814](https://github.com/rust-lang/rust-clippy/pull/4814)
  * [`zst_offset`] [#4816](https://github.com/rust-lang/rust-clippy/pull/4816)
  * [`as_conversions`] [#4821](https://github.com/rust-lang/rust-clippy/pull/4821)
  * [`missing_errors_doc`] [#4884](https://github.com/rust-lang/rust-clippy/pull/4884)
  * [`transmute_float_to_int`] [#4889](https://github.com/rust-lang/rust-clippy/pull/4889)
* Remove plugin interface, see
  [Inside Rust Blog](https://blog.rust-lang.org/inside-rust/2019/11/04/Clippy-removes-plugin-interface.html) for
  details [#4714](https://github.com/rust-lang/rust-clippy/pull/4714)
* Move [`use_self`] to nursery group [#4863](https://github.com/rust-lang/rust-clippy/pull/4863)
* Deprecate [`into_iter_on_array`] [#4788](https://github.com/rust-lang/rust-clippy/pull/4788)
* Expand [`string_lit_as_bytes`] to also trigger when literal has escapes
  [#4808](https://github.com/rust-lang/rust-clippy/pull/4808)
* Fix false positive in `comparison_chain` [#4842](https://github.com/rust-lang/rust-clippy/pull/4842)
* Fix false positive in `while_immutable_condition` [#4730](https://github.com/rust-lang/rust-clippy/pull/4730)
* Fix false positive in `explicit_counter_loop` [#4803](https://github.com/rust-lang/rust-clippy/pull/4803)
* Fix false positive in `must_use_candidate` [#4794](https://github.com/rust-lang/rust-clippy/pull/4794)
* Fix false positive in `print_with_newline` and `write_with_newline`
  [#4769](https://github.com/rust-lang/rust-clippy/pull/4769)
* Fix false positive in `derive_hash_xor_eq` [#4766](https://github.com/rust-lang/rust-clippy/pull/4766)
* Fix false positive in `missing_inline_in_public_items` [#4870](https://github.com/rust-lang/rust-clippy/pull/4870)
* Fix false positive in `string_add` [#4880](https://github.com/rust-lang/rust-clippy/pull/4880)
* Fix false positive in `float_arithmetic` [#4851](https://github.com/rust-lang/rust-clippy/pull/4851)
* Fix false positive in `cast_sign_loss` [#4883](https://github.com/rust-lang/rust-clippy/pull/4883)
* Fix false positive in `manual_swap` [#4877](https://github.com/rust-lang/rust-clippy/pull/4877)
* Fix ICEs occurring while checking some block expressions [#4772](https://github.com/rust-lang/rust-clippy/pull/4772)
* Fix ICE in `use_self` [#4776](https://github.com/rust-lang/rust-clippy/pull/4776)
* Fix ICEs related to `const_generics` [#4780](https://github.com/rust-lang/rust-clippy/pull/4780)
* Display help when running `clippy-driver` without arguments, instead of ICEing
  [#4810](https://github.com/rust-lang/rust-clippy/pull/4810)
* Clippy has its own ICE message now [#4588](https://github.com/rust-lang/rust-clippy/pull/4588)
* Show deprecated lints in the documentation again [#4757](https://github.com/rust-lang/rust-clippy/pull/4757)
* Improve Documentation by adding positive examples to some lints
  [#4832](https://github.com/rust-lang/rust-clippy/pull/4832)

## Rust 1.40

Released 2019-12-19

[4e7e71b...c8e3cfb](https://github.com/rust-lang/rust-clippy/compare/4e7e71b...c8e3cfb)

* New Lints:
  * [`unneeded_wildcard_pattern`] [#4537](https://github.com/rust-lang/rust-clippy/pull/4537)
  * [`needless_doctest_main`] [#4603](https://github.com/rust-lang/rust-clippy/pull/4603)
  * [`suspicious_unary_op_formatting`] [#4615](https://github.com/rust-lang/rust-clippy/pull/4615)
  * [`debug_assert_with_mut_call`] [#4680](https://github.com/rust-lang/rust-clippy/pull/4680)
  * [`unused_self`] [#4619](https://github.com/rust-lang/rust-clippy/pull/4619)
  * [`inefficient_to_string`] [#4683](https://github.com/rust-lang/rust-clippy/pull/4683)
  * [`must_use_unit`] [#4560](https://github.com/rust-lang/rust-clippy/pull/4560)
  * [`must_use_candidate`] [#4560](https://github.com/rust-lang/rust-clippy/pull/4560)
  * [`double_must_use`] [#4560](https://github.com/rust-lang/rust-clippy/pull/4560)
  * [`comparison_chain`] [#4569](https://github.com/rust-lang/rust-clippy/pull/4569)
  * [`unsound_collection_transmute`] [#4592](https://github.com/rust-lang/rust-clippy/pull/4592)
  * [`panic`] [#4657](https://github.com/rust-lang/rust-clippy/pull/4657)
  * [`unreachable`] [#4657](https://github.com/rust-lang/rust-clippy/pull/4657)
  * [`todo`] [#4657](https://github.com/rust-lang/rust-clippy/pull/4657)
  * `option_expect_used` [#4657](https://github.com/rust-lang/rust-clippy/pull/4657)
  * `result_expect_used` [#4657](https://github.com/rust-lang/rust-clippy/pull/4657)
* Move `redundant_clone` to perf group [#4509](https://github.com/rust-lang/rust-clippy/pull/4509)
* Move `manual_mul_add` to nursery group [#4736](https://github.com/rust-lang/rust-clippy/pull/4736)
* Expand `unit_cmp` to also work with `assert_eq!`, `debug_assert_eq!`, `assert_ne!` and `debug_assert_ne!` [#4613](https://github.com/rust-lang/rust-clippy/pull/4613)
* Expand `integer_arithmetic` to also detect mutating arithmetic like `+=` [#4585](https://github.com/rust-lang/rust-clippy/pull/4585)
* Fix false positive in `nonminimal_bool` [#4568](https://github.com/rust-lang/rust-clippy/pull/4568)
* Fix false positive in `missing_safety_doc` [#4611](https://github.com/rust-lang/rust-clippy/pull/4611)
* Fix false positive in `cast_sign_loss` [#4614](https://github.com/rust-lang/rust-clippy/pull/4614)
* Fix false positive in `redundant_clone` [#4509](https://github.com/rust-lang/rust-clippy/pull/4509)
* Fix false positive in `try_err` [#4721](https://github.com/rust-lang/rust-clippy/pull/4721)
* Fix false positive in `toplevel_ref_arg` [#4570](https://github.com/rust-lang/rust-clippy/pull/4570)
* Fix false positive in `multiple_inherent_impl` [#4593](https://github.com/rust-lang/rust-clippy/pull/4593)
* Improve more suggestions and tests in preparation for the unstable `cargo fix --clippy` [#4575](https://github.com/rust-lang/rust-clippy/pull/4575)
* Improve suggestion for `zero_ptr` [#4599](https://github.com/rust-lang/rust-clippy/pull/4599)
* Improve suggestion for `explicit_counter_loop` [#4691](https://github.com/rust-lang/rust-clippy/pull/4691)
* Improve suggestion for `mul_add` [#4602](https://github.com/rust-lang/rust-clippy/pull/4602)
* Improve suggestion for `assertions_on_constants` [#4635](https://github.com/rust-lang/rust-clippy/pull/4635)
* Fix ICE in `use_self` [#4671](https://github.com/rust-lang/rust-clippy/pull/4671)
* Fix ICE when encountering const casts [#4590](https://github.com/rust-lang/rust-clippy/pull/4590)

## Rust 1.39

Released 2019-11-07

[3aea860...4e7e71b](https://github.com/rust-lang/rust-clippy/compare/3aea860...4e7e71b)

* New Lints:
  * [`uninit_assumed_init`] [#4479](https://github.com/rust-lang/rust-clippy/pull/4479)
  * [`flat_map_identity`] [#4231](https://github.com/rust-lang/rust-clippy/pull/4231)
  * [`missing_safety_doc`] [#4535](https://github.com/rust-lang/rust-clippy/pull/4535)
  * [`mem_replace_with_uninit`] [#4511](https://github.com/rust-lang/rust-clippy/pull/4511)
  * [`suspicious_map`] [#4394](https://github.com/rust-lang/rust-clippy/pull/4394)
  * `option_and_then_some` [#4386](https://github.com/rust-lang/rust-clippy/pull/4386)
  * [`manual_saturating_arithmetic`] [#4498](https://github.com/rust-lang/rust-clippy/pull/4498)
* Deprecate `unused_collect` lint. This is fully covered by rustc's `#[must_use]` on `collect` [#4348](https://github.com/rust-lang/rust-clippy/pull/4348)
* Move `type_repetition_in_bounds` to pedantic group [#4403](https://github.com/rust-lang/rust-clippy/pull/4403)
* Move `cast_lossless` to pedantic group [#4539](https://github.com/rust-lang/rust-clippy/pull/4539)
* `temporary_cstring_as_ptr` now catches more cases [#4425](https://github.com/rust-lang/rust-clippy/pull/4425)
* `use_self` now works in constructors, too [#4525](https://github.com/rust-lang/rust-clippy/pull/4525)
* `cargo_common_metadata` now checks for license files [#4518](https://github.com/rust-lang/rust-clippy/pull/4518)
* `cognitive_complexity` now includes the measured complexity in the warning message [#4469](https://github.com/rust-lang/rust-clippy/pull/4469)
* Fix false positives in `block_in_if_*` lints [#4458](https://github.com/rust-lang/rust-clippy/pull/4458)
* Fix false positive in `cast_lossless` [#4473](https://github.com/rust-lang/rust-clippy/pull/4473)
* Fix false positive in `clone_on_copy` [#4411](https://github.com/rust-lang/rust-clippy/pull/4411)
* Fix false positive in `deref_addrof` [#4487](https://github.com/rust-lang/rust-clippy/pull/4487)
* Fix false positive in `too_many_lines` [#4490](https://github.com/rust-lang/rust-clippy/pull/4490)
* Fix false positive in `new_ret_no_self` [#4365](https://github.com/rust-lang/rust-clippy/pull/4365)
* Fix false positive in `manual_swap` [#4478](https://github.com/rust-lang/rust-clippy/pull/4478)
* Fix false positive in `missing_const_for_fn` [#4450](https://github.com/rust-lang/rust-clippy/pull/4450)
* Fix false positive in `extra_unused_lifetimes` [#4477](https://github.com/rust-lang/rust-clippy/pull/4477)
* Fix false positive in `inherent_to_string` [#4460](https://github.com/rust-lang/rust-clippy/pull/4460)
* Fix false positive in `map_entry` [#4495](https://github.com/rust-lang/rust-clippy/pull/4495)
* Fix false positive in `unused_unit` [#4445](https://github.com/rust-lang/rust-clippy/pull/4445)
* Fix false positive in `redundant_pattern` [#4489](https://github.com/rust-lang/rust-clippy/pull/4489)
* Fix false positive in `wrong_self_convention` [#4369](https://github.com/rust-lang/rust-clippy/pull/4369)
* Improve various suggestions and tests in preparation for the unstable `cargo fix --clippy` [#4558](https://github.com/rust-lang/rust-clippy/pull/4558)
* Improve suggestions for `redundant_pattern_matching` [#4352](https://github.com/rust-lang/rust-clippy/pull/4352)
* Improve suggestions for `explicit_write` [#4544](https://github.com/rust-lang/rust-clippy/pull/4544)
* Improve suggestion for `or_fun_call` [#4522](https://github.com/rust-lang/rust-clippy/pull/4522)
* Improve suggestion for `match_as_ref` [#4446](https://github.com/rust-lang/rust-clippy/pull/4446)
* Improve suggestion for `unnecessary_fold_span` [#4382](https://github.com/rust-lang/rust-clippy/pull/4382)
* Add suggestions for `unseparated_literal_suffix` [#4401](https://github.com/rust-lang/rust-clippy/pull/4401)
* Add suggestions for `char_lit_as_u8` [#4418](https://github.com/rust-lang/rust-clippy/pull/4418)

## Rust 1.38

Released 2019-09-26

[e3cb40e...3aea860](https://github.com/rust-lang/rust-clippy/compare/e3cb40e...3aea860)

* New Lints:
  * [`main_recursion`] [#4203](https://github.com/rust-lang/rust-clippy/pull/4203)
  * [`inherent_to_string`] [#4259](https://github.com/rust-lang/rust-clippy/pull/4259)
  * [`inherent_to_string_shadow_display`] [#4259](https://github.com/rust-lang/rust-clippy/pull/4259)
  * [`type_repetition_in_bounds`] [#3766](https://github.com/rust-lang/rust-clippy/pull/3766)
  * [`try_err`] [#4222](https://github.com/rust-lang/rust-clippy/pull/4222)
* Move `{unnnecessary,panicking}_unwrap` out of nursery [#4307](https://github.com/rust-lang/rust-clippy/pull/4307)
* Extend the `use_self` lint to suggest uses of `Self::Variant` [#4308](https://github.com/rust-lang/rust-clippy/pull/4308)
* Improve suggestion for needless return [#4262](https://github.com/rust-lang/rust-clippy/pull/4262)
* Add auto-fixable suggestion for `let_unit` [#4337](https://github.com/rust-lang/rust-clippy/pull/4337)
* Fix false positive in `pub_enum_variant_names` and `enum_variant_names` [#4345](https://github.com/rust-lang/rust-clippy/pull/4345)
* Fix false positive in `cast_ptr_alignment` [#4257](https://github.com/rust-lang/rust-clippy/pull/4257)
* Fix false positive in `string_lit_as_bytes` [#4233](https://github.com/rust-lang/rust-clippy/pull/4233)
* Fix false positive in `needless_lifetimes` [#4266](https://github.com/rust-lang/rust-clippy/pull/4266)
* Fix false positive in `float_cmp` [#4275](https://github.com/rust-lang/rust-clippy/pull/4275)
* Fix false positives in `needless_return` [#4274](https://github.com/rust-lang/rust-clippy/pull/4274)
* Fix false negative in `match_same_arms` [#4246](https://github.com/rust-lang/rust-clippy/pull/4246)
* Fix incorrect suggestion for `needless_bool` [#4335](https://github.com/rust-lang/rust-clippy/pull/4335)
* Improve suggestion for `cast_ptr_alignment` [#4257](https://github.com/rust-lang/rust-clippy/pull/4257)
* Improve suggestion for `single_char_literal` [#4361](https://github.com/rust-lang/rust-clippy/pull/4361)
* Improve suggestion for `len_zero` [#4314](https://github.com/rust-lang/rust-clippy/pull/4314)
* Fix ICE in `implicit_hasher` [#4268](https://github.com/rust-lang/rust-clippy/pull/4268)
* Fix allow bug in `trivially_copy_pass_by_ref` [#4250](https://github.com/rust-lang/rust-clippy/pull/4250)

## Rust 1.37

Released 2019-08-15

[082cfa7...e3cb40e](https://github.com/rust-lang/rust-clippy/compare/082cfa7...e3cb40e)

* New Lints:
  * [`checked_conversions`] [#4088](https://github.com/rust-lang/rust-clippy/pull/4088)
  * [`get_last_with_len`] [#3832](https://github.com/rust-lang/rust-clippy/pull/3832)
  * [`integer_division`] [#4195](https://github.com/rust-lang/rust-clippy/pull/4195)
* Renamed Lint: `const_static_lifetime` is now called [`redundant_static_lifetimes`].
  The lint now covers statics in addition to consts [#4162](https://github.com/rust-lang/rust-clippy/pull/4162)
* [`match_same_arms`] now warns for all identical arms, instead of only the first one [#4102](https://github.com/rust-lang/rust-clippy/pull/4102)
* [`needless_return`] now works with void functions [#4220](https://github.com/rust-lang/rust-clippy/pull/4220)
* Fix false positive in [`redundant_closure`] [#4190](https://github.com/rust-lang/rust-clippy/pull/4190)
* Fix false positive in [`useless_attribute`] [#4107](https://github.com/rust-lang/rust-clippy/pull/4107)
* Fix incorrect suggestion for [`float_cmp`] [#4214](https://github.com/rust-lang/rust-clippy/pull/4214)
* Add suggestions for [`print_with_newline`] and [`write_with_newline`] [#4136](https://github.com/rust-lang/rust-clippy/pull/4136)
* Improve suggestions for `option_map_unwrap_or_else` and `result_map_unwrap_or_else` [#4164](https://github.com/rust-lang/rust-clippy/pull/4164)
* Improve suggestions for [`non_ascii_literal`] [#4119](https://github.com/rust-lang/rust-clippy/pull/4119)
* Improve diagnostics for [`let_and_return`] [#4137](https://github.com/rust-lang/rust-clippy/pull/4137)
* Improve diagnostics for [`trivially_copy_pass_by_ref`] [#4071](https://github.com/rust-lang/rust-clippy/pull/4071)
* Add macro check for [`unreadable_literal`] [#4099](https://github.com/rust-lang/rust-clippy/pull/4099)

## Rust 1.36

Released 2019-07-04

[eb9f9b1...082cfa7](https://github.com/rust-lang/rust-clippy/compare/eb9f9b1...082cfa7)

* New lints: [`find_map`], [`filter_map_next`] [#4039](https://github.com/rust-lang/rust-clippy/pull/4039)
* New lint: [`path_buf_push_overwrite`] [#3954](https://github.com/rust-lang/rust-clippy/pull/3954)
* Move `path_buf_push_overwrite` to the nursery [#4013](https://github.com/rust-lang/rust-clippy/pull/4013)
* Split [`redundant_closure`] into [`redundant_closure`] and [`redundant_closure_for_method_calls`] [#4110](https://github.com/rust-lang/rust-clippy/pull/4101)
* Allow allowing of [`toplevel_ref_arg`] lint [#4007](https://github.com/rust-lang/rust-clippy/pull/4007)
* Fix false negative in [`or_fun_call`] pertaining to nested constructors [#4084](https://github.com/rust-lang/rust-clippy/pull/4084)
* Fix false positive in [`or_fun_call`] pertaining to enum variant constructors [#4018](https://github.com/rust-lang/rust-clippy/pull/4018)
* Fix false positive in [`useless_let_if_seq`] pertaining to interior mutability [#4035](https://github.com/rust-lang/rust-clippy/pull/4035)
* Fix false positive in [`redundant_closure`] pertaining to non-function types [#4008](https://github.com/rust-lang/rust-clippy/pull/4008)
* Fix false positive in [`let_and_return`] pertaining to attributes on `let`s [#4024](https://github.com/rust-lang/rust-clippy/pull/4024)
* Fix false positive in [`module_name_repetitions`] lint pertaining to attributes [#4006](https://github.com/rust-lang/rust-clippy/pull/4006)
* Fix false positive on [`assertions_on_constants`] pertaining to `debug_assert!` [#3989](https://github.com/rust-lang/rust-clippy/pull/3989)
* Improve suggestion in [`map_clone`] to suggest `.copied()` where applicable  [#3970](https://github.com/rust-lang/rust-clippy/pull/3970) [#4043](https://github.com/rust-lang/rust-clippy/pull/4043)
* Improve suggestion for [`search_is_some`] [#4049](https://github.com/rust-lang/rust-clippy/pull/4049)
* Improve suggestion applicability for [`naive_bytecount`] [#3984](https://github.com/rust-lang/rust-clippy/pull/3984)
* Improve suggestion applicability for [`while_let_loop`] [#3975](https://github.com/rust-lang/rust-clippy/pull/3975)
* Improve diagnostics for [`too_many_arguments`] [#4053](https://github.com/rust-lang/rust-clippy/pull/4053)
* Improve diagnostics for [`cast_lossless`] [#4021](https://github.com/rust-lang/rust-clippy/pull/4021)
* Deal with macro checks in desugarings better [#4082](https://github.com/rust-lang/rust-clippy/pull/4082)
* Add macro check for [`unnecessary_cast`]  [#4026](https://github.com/rust-lang/rust-clippy/pull/4026)
* Remove [`approx_constant`]'s documentation's "Known problems" section. [#4027](https://github.com/rust-lang/rust-clippy/pull/4027)
* Fix ICE in [`suspicious_else_formatting`] [#3960](https://github.com/rust-lang/rust-clippy/pull/3960)
* Fix ICE in [`decimal_literal_representation`] [#3931](https://github.com/rust-lang/rust-clippy/pull/3931)


## Rust 1.35

Released 2019-05-20

[1fac380..37f5c1e](https://github.com/rust-lang/rust-clippy/compare/1fac380...37f5c1e)

* New lint: [`drop_bounds`] to detect `T: Drop` bounds
* Split [`redundant_closure`] into [`redundant_closure`] and [`redundant_closure_for_method_calls`] [#4110](https://github.com/rust-lang/rust-clippy/pull/4101)
* Rename `cyclomatic_complexity` to [`cognitive_complexity`], start work on making lint more practical for Rust code
* Move [`get_unwrap`] to the restriction category
* Improve suggestions for [`iter_cloned_collect`]
* Improve suggestions for [`cast_lossless`] to suggest suffixed literals
* Fix false positives in [`print_with_newline`] and [`write_with_newline`] pertaining to raw strings
* Fix false positive in [`needless_range_loop`] pertaining to structs without a `.iter()`
* Fix false positive in [`bool_comparison`] pertaining to non-bool types
* Fix false positive in [`redundant_closure`] pertaining to differences in borrows
* Fix false positive in `option_map_unwrap_or` on non-copy types
* Fix false positives in [`missing_const_for_fn`] pertaining to macros and trait method impls
* Fix false positive in [`needless_pass_by_value`] pertaining to procedural macros
* Fix false positive in [`needless_continue`] pertaining to loop labels
* Fix false positive for [`boxed_local`] pertaining to arguments moved into closures
* Fix false positive for [`use_self`] in nested functions
* Fix suggestion for [`expect_fun_call`] (https://github.com/rust-lang/rust-clippy/pull/3846)
* Fix suggestion for [`explicit_counter_loop`] to deal with parenthesizing range variables
* Fix suggestion for [`single_char_pattern`] to correctly escape single quotes
* Avoid triggering [`redundant_closure`] in macros
* ICE fixes: [#3805](https://github.com/rust-lang/rust-clippy/pull/3805), [#3772](https://github.com/rust-lang/rust-clippy/pull/3772), [#3741](https://github.com/rust-lang/rust-clippy/pull/3741)

## Rust 1.34

Released 2019-04-10

[1b89724...1fac380](https://github.com/rust-lang/rust-clippy/compare/1b89724...1fac380)

* New lint: [`assertions_on_constants`] to detect for example `assert!(true)`
* New lint: [`dbg_macro`] to detect uses of the `dbg!` macro
* New lint: [`missing_const_for_fn`] that can suggest functions to be made `const`
* New lint: [`too_many_lines`] to detect functions with excessive LOC. It can be
  configured using the `too-many-lines-threshold` configuration.
* New lint: [`wildcard_enum_match_arm`] to check for wildcard enum matches using `_`
* Expand `redundant_closure` to also work for methods (not only functions)
* Fix ICEs in `vec_box`, `needless_pass_by_value` and `implicit_hasher`
* Fix false positive in `cast_sign_loss`
* Fix false positive in `integer_arithmetic`
* Fix false positive in `unit_arg`
* Fix false positives in `implicit_return`
* Add suggestion to `explicit_write`
* Improve suggestions for `question_mark` lint
* Fix incorrect suggestion for `cast_lossless`
* Fix incorrect suggestion for `expect_fun_call`
* Fix incorrect suggestion for `needless_bool`
* Fix incorrect suggestion for `needless_range_loop`
* Fix incorrect suggestion for `use_self`
* Fix incorrect suggestion for `while_let_on_iterator`
* Clippy is now slightly easier to invoke in non-cargo contexts. See
  [#3665][pull3665] for more details.
* We now have [improved documentation][adding_lints] on how to add new lints

## Rust 1.33

Released 2019-02-26

[b2601be...1b89724](https://github.com/rust-lang/rust-clippy/compare/b2601be...1b89724)

* New lints: [`implicit_return`], [`vec_box`], [`cast_ref_to_mut`]
* The `rust-clippy` repository is now part of the `rust-lang` org.
* Rename `stutter` to `module_name_repetitions`
* Merge `new_without_default_derive` into `new_without_default` lint
* Move `large_digit_groups` from `style` group to `pedantic`
* Expand `bool_comparison` to check for `<`, `<=`, `>`, `>=`, and `!=`
  comparisons against booleans
* Expand `no_effect` to detect writes to constants such as `A_CONST.field = 2`
* Expand `redundant_clone` to work on struct fields
* Expand `suspicious_else_formatting` to detect `if .. {..} {..}`
* Expand `use_self` to work on tuple structs and also in local macros
* Fix ICE in `result_map_unit_fn` and `option_map_unit_fn`
* Fix false positives in `implicit_return`
* Fix false positives in `use_self`
* Fix false negative in `clone_on_copy`
* Fix false positive in `doc_markdown`
* Fix false positive in `empty_loop`
* Fix false positive in `if_same_then_else`
* Fix false positive in `infinite_iter`
* Fix false positive in `question_mark`
* Fix false positive in `useless_asref`
* Fix false positive in `wildcard_dependencies`
* Fix false positive in `write_with_newline`
* Add suggestion to `explicit_write`
* Improve suggestions for `question_mark` lint
* Fix incorrect suggestion for `get_unwrap`

## Rust 1.32

Released 2019-01-17

[2e26fdc2...b2601be](https://github.com/rust-lang/rust-clippy/compare/2e26fdc2...b2601be)

* New lints: [`slow_vector_initialization`], [`mem_discriminant_non_enum`],
  [`redundant_clone`], [`wildcard_dependencies`],
  [`into_iter_on_ref`], [`into_iter_on_array`], [`deprecated_cfg_attr`],
  [`mem_discriminant_non_enum`], [`cargo_common_metadata`]
* Add support for `u128` and `i128` to integer related lints
* Add float support to `mistyped_literal_suffixes`
* Fix false positives in `use_self`
* Fix false positives in `missing_comma`
* Fix false positives in `new_ret_no_self`
* Fix false positives in `possible_missing_comma`
* Fix false positive in `integer_arithmetic` in constant items
* Fix false positive in `needless_borrow`
* Fix false positive in `out_of_bounds_indexing`
* Fix false positive in `new_without_default_derive`
* Fix false positive in `string_lit_as_bytes`
* Fix false negative in `out_of_bounds_indexing`
* Fix false negative in `use_self`. It will now also check existential types
* Fix incorrect suggestion for `redundant_closure_call`
* Fix various suggestions that contained expanded macros
* Fix `bool_comparison` triggering 3 times on on on the same code
* Expand `trivially_copy_pass_by_ref` to work on trait methods
* Improve suggestion for `needless_range_loop`
* Move `needless_pass_by_value` from `pedantic` group to `style`

## Rust 1.31

Released 2018-12-06

[125907ad..2e26fdc2](https://github.com/rust-lang/rust-clippy/compare/125907ad..2e26fdc2)

* Clippy has been relicensed under a dual MIT / Apache license.
  See [#3093](https://github.com/rust-lang/rust-clippy/issues/3093) for more
  information.
* With Rust 1.31, Clippy is no longer available via crates.io. The recommended
  installation method is via `rustup component add clippy`.
* New lints: [`redundant_pattern_matching`], [`unnecessary_filter_map`],
  [`unused_unit`], [`map_flatten`], [`mem_replace_option_with_none`]
* Fix ICE in `if_let_redundant_pattern_matching`
* Fix ICE in `needless_pass_by_value` when encountering a generic function
  argument with a lifetime parameter
* Fix ICE in `needless_range_loop`
* Fix ICE in `single_char_pattern` when encountering a constant value
* Fix false positive in `assign_op_pattern`
* Fix false positive in `boxed_local` on trait implementations
* Fix false positive in `cmp_owned`
* Fix false positive in `collapsible_if` when conditionals have comments
* Fix false positive in `double_parens`
* Fix false positive in `excessive_precision`
* Fix false positive in `explicit_counter_loop`
* Fix false positive in `fn_to_numeric_cast_with_truncation`
* Fix false positive in `map_clone`
* Fix false positive in `new_ret_no_self`
* Fix false positive in `new_without_default` when `new` is unsafe
* Fix false positive in `type_complexity` when using extern types
* Fix false positive in `useless_format`
* Fix false positive in `wrong_self_convention`
* Fix incorrect suggestion for `excessive_precision`
* Fix incorrect suggestion for `expect_fun_call`
* Fix incorrect suggestion for `get_unwrap`
* Fix incorrect suggestion for `useless_format`
* `fn_to_numeric_cast_with_truncation` lint can be disabled again
* Improve suggestions for `manual_memcpy`
* Improve help message for `needless_lifetimes`

## Rust 1.30

Released 2018-10-25

[14207503...125907ad](https://github.com/rust-lang/rust-clippy/compare/14207503...125907ad)

* Deprecate `assign_ops` lint
* New lints: [`mistyped_literal_suffixes`], [`ptr_offset_with_cast`],
  [`needless_collect`], [`copy_iterator`]
* `cargo clippy -V` now includes the Clippy commit hash of the Rust
  Clippy component
* Fix ICE in `implicit_hasher`
* Fix ICE when encountering `println!("{}" a);`
* Fix ICE when encountering a macro call in match statements
* Fix false positive in `default_trait_access`
* Fix false positive in `trivially_copy_pass_by_ref`
* Fix false positive in `similar_names`
* Fix false positive in `redundant_field_name`
* Fix false positive in `expect_fun_call`
* Fix false negative in `identity_conversion`
* Fix false negative in `explicit_counter_loop`
* Fix `range_plus_one` suggestion and false negative
* `print_with_newline` / `write_with_newline`: don't warn about string with several `\n`s in them
* Fix `useless_attribute` to also whitelist `unused_extern_crates`
* Fix incorrect suggestion for `single_char_pattern`
* Improve suggestion for `identity_conversion` lint
* Move `explicit_iter_loop` and `explicit_into_iter_loop` from `style` group to `pedantic`
* Move `range_plus_one` and `range_minus_one` from `nursery` group to `complexity`
* Move `shadow_unrelated` from `restriction` group to `pedantic`
* Move `indexing_slicing` from `pedantic` group to `restriction`

## Rust 1.29

Released 2018-09-13

[v0.0.212...14207503](https://github.com/rust-lang/rust-clippy/compare/v0.0.212...14207503)

* :tada: :tada: **Rust 1.29 is the first stable Rust that includes a bundled Clippy** :tada:
  :tada:
  You can now run `rustup component add clippy-preview` and then `cargo
  clippy` to run Clippy. This should put an end to the continuous nightly
  upgrades for Clippy users.
* Clippy now follows the Rust versioning scheme instead of its own
* Fix ICE when encountering a `while let (..) = x.iter()` construct
* Fix false positives in `use_self`
* Fix false positive in `trivially_copy_pass_by_ref`
* Fix false positive in `useless_attribute` lint
* Fix false positive in `print_literal`
* Fix `use_self` regressions
* Improve lint message for `neg_cmp_op_on_partial_ord`
* Improve suggestion highlight for `single_char_pattern`
* Improve suggestions for various print/write macro lints
* Improve website header

## 0.0.212 (2018-07-10)
* Rustup to *rustc 1.29.0-nightly (e06c87544 2018-07-06)*

## 0.0.211
* Rustup to *rustc 1.28.0-nightly (e3bf634e0 2018-06-28)*

## 0.0.210
* Rustup to *rustc 1.28.0-nightly (01cc982e9 2018-06-24)*

## 0.0.209
* Rustup to *rustc 1.28.0-nightly (523097979 2018-06-18)*

## 0.0.208
* Rustup to *rustc 1.28.0-nightly (86a8f1a63 2018-06-17)*

## 0.0.207
* Rustup to *rustc 1.28.0-nightly (2a0062974 2018-06-09)*

## 0.0.206
* Rustup to *rustc 1.28.0-nightly (5bf68db6e 2018-05-28)*

## 0.0.205
* Rustup to *rustc 1.28.0-nightly (990d8aa74 2018-05-25)*
* Rename `unused_lifetimes` to `extra_unused_lifetimes` because of naming conflict with new rustc lint

## 0.0.204
* Rustup to *rustc 1.28.0-nightly (71e87be38 2018-05-22)*

## 0.0.203
* Rustup to *rustc 1.28.0-nightly (a3085756e 2018-05-19)*
* Clippy attributes are now of the form `clippy::cyclomatic_complexity` instead of `clippy(cyclomatic_complexity)`

## 0.0.202
* Rustup to *rustc 1.28.0-nightly (952f344cd 2018-05-18)*

## 0.0.201
* Rustup to *rustc 1.27.0-nightly (2f2a11dfc 2018-05-16)*

## 0.0.200
* Rustup to *rustc 1.27.0-nightly (9fae15374 2018-05-13)*

## 0.0.199
* Rustup to *rustc 1.27.0-nightly (ff2ac35db 2018-05-12)*

## 0.0.198
* Rustup to *rustc 1.27.0-nightly (acd3871ba 2018-05-10)*

## 0.0.197
* Rustup to *rustc 1.27.0-nightly (428ea5f6b 2018-05-06)*

## 0.0.196
* Rustup to *rustc 1.27.0-nightly (e82261dfb 2018-05-03)*

## 0.0.195
* Rustup to *rustc 1.27.0-nightly (ac3c2288f 2018-04-18)*

## 0.0.194
* Rustup to *rustc 1.27.0-nightly (bd40cbbe1 2018-04-14)*
* New lints: [`cast_ptr_alignment`], [`transmute_ptr_to_ptr`], [`write_literal`], [`write_with_newline`], [`writeln_empty_string`]

## 0.0.193
* Rustup to *rustc 1.27.0-nightly (eeea94c11 2018-04-06)*

## 0.0.192
* Rustup to *rustc 1.27.0-nightly (fb44b4c0e 2018-04-04)*
* New lint: [`print_literal`]

## 0.0.191
* Rustup to *rustc 1.26.0-nightly (ae544ee1c 2018-03-29)*
* Lint audit; categorize lints as style, correctness, complexity, pedantic, nursery, restriction.

## 0.0.190
* Fix a bunch of intermittent cargo bugs

## 0.0.189
* Rustup to *rustc 1.26.0-nightly (5508b2714 2018-03-18)*

## 0.0.188
* Rustup to *rustc 1.26.0-nightly (392645394 2018-03-15)*
* New lint: [`while_immutable_condition`]

## 0.0.187
* Rustup to *rustc 1.26.0-nightly (322d7f7b9 2018-02-25)*
* New lints: [`redundant_field_names`], [`suspicious_arithmetic_impl`], [`suspicious_op_assign_impl`]

## 0.0.186
* Rustup to *rustc 1.25.0-nightly (0c6091fbd 2018-02-04)*
* Various false positive fixes

## 0.0.185
* Rustup to *rustc 1.25.0-nightly (56733bc9f 2018-02-01)*
* New lint: [`question_mark`]

## 0.0.184
* Rustup to *rustc 1.25.0-nightly (90eb44a58 2018-01-29)*
* New lints: [`double_comparisons`], [`empty_line_after_outer_attr`]

## 0.0.183
* Rustup to *rustc 1.25.0-nightly (21882aad7 2018-01-28)*
* New lint: [`misaligned_transmute`]

## 0.0.182
* Rustup to *rustc 1.25.0-nightly (a0dcecff9 2018-01-24)*
* New lint: [`decimal_literal_representation`]

## 0.0.181
* Rustup to *rustc 1.25.0-nightly (97520ccb1 2018-01-21)*
* New lints: [`else_if_without_else`], [`option_option`], [`unit_arg`], [`unnecessary_fold`]
* Removed `unit_expr`
* Various false positive fixes for [`needless_pass_by_value`]

## 0.0.180
* Rustup to *rustc 1.25.0-nightly (3f92e8d89 2018-01-14)*

## 0.0.179
* Rustup to *rustc 1.25.0-nightly (61452e506 2018-01-09)*

## 0.0.178
* Rustup to *rustc 1.25.0-nightly (ee220daca 2018-01-07)*

## 0.0.177
* Rustup to *rustc 1.24.0-nightly (250b49205 2017-12-21)*
* New lint: [`match_as_ref`]

## 0.0.176
* Rustup to *rustc 1.24.0-nightly (0077d128d 2017-12-14)*

## 0.0.175
* Rustup to *rustc 1.24.0-nightly (bb42071f6 2017-12-01)*

## 0.0.174
* Rustup to *rustc 1.23.0-nightly (63739ab7b 2017-11-21)*

## 0.0.173
* Rustup to *rustc 1.23.0-nightly (33374fa9d 2017-11-20)*

## 0.0.172
* Rustup to *rustc 1.23.0-nightly (d0f8e2913 2017-11-16)*

## 0.0.171
* Rustup to *rustc 1.23.0-nightly (ff0f5de3b 2017-11-14)*

## 0.0.170
* Rustup to *rustc 1.23.0-nightly (d6b06c63a 2017-11-09)*

## 0.0.169
* Rustup to *rustc 1.23.0-nightly (3b82e4c74 2017-11-05)*
* New lints: [`just_underscores_and_digits`], `result_map_unwrap_or_else`, [`transmute_bytes_to_str`]

## 0.0.168
* Rustup to *rustc 1.23.0-nightly (f0fe716db 2017-10-30)*

## 0.0.167
* Rustup to *rustc 1.23.0-nightly (90ef3372e 2017-10-29)*
* New lints: `const_static_lifetime`, [`erasing_op`], [`fallible_impl_from`], [`println_empty_string`], [`useless_asref`]

## 0.0.166
* Rustup to *rustc 1.22.0-nightly (b7960878b 2017-10-18)*
* New lints: [`explicit_write`], `identity_conversion`, [`implicit_hasher`], [`invalid_ref`], [`option_map_or_none`],
  [`range_minus_one`], [`range_plus_one`], [`transmute_int_to_bool`], [`transmute_int_to_char`],
  [`transmute_int_to_float`]

## 0.0.165
* Rust upgrade to rustc 1.22.0-nightly (0e6f4cf51 2017-09-27)
* New lint: [`mut_range_bound`]

## 0.0.164
* Update to *rustc 1.22.0-nightly (6c476ce46 2017-09-25)*
* New lint: [`int_plus_one`]

## 0.0.163
* Update to *rustc 1.22.0-nightly (14039a42a 2017-09-22)*

## 0.0.162
* Update to *rustc 1.22.0-nightly (0701b37d9 2017-09-18)*
* New lint: [`chars_last_cmp`]
* Improved suggestions for [`needless_borrow`], [`ptr_arg`],

## 0.0.161
* Update to *rustc 1.22.0-nightly (539f2083d 2017-09-13)*

## 0.0.160
* Update to *rustc 1.22.0-nightly (dd08c3070 2017-09-12)*

## 0.0.159
* Update to *rustc 1.22.0-nightly (eba374fb2 2017-09-11)*
* New lint: [`clone_on_ref_ptr`]

## 0.0.158
* New lint: [`manual_memcpy`]
* [`cast_lossless`] no longer has redundant parentheses in its suggestions
* Update to *rustc 1.22.0-nightly (dead08cb3 2017-09-08)*

## 0.0.157 - 2017-09-04
* Update to *rustc 1.22.0-nightly (981ce7d8d 2017-09-03)*
* New lint: `unit_expr`

## 0.0.156 - 2017-09-03
* Update to *rustc 1.22.0-nightly (744dd6c1d 2017-09-02)*

## 0.0.155
* Update to *rustc 1.21.0-nightly (c11f689d2 2017-08-29)*
* New lint: [`infinite_iter`], [`maybe_infinite_iter`], [`cast_lossless`]

## 0.0.154
* Update to *rustc 1.21.0-nightly (2c0558f63 2017-08-24)*
* Fix [`use_self`] triggering inside derives
* Add support for linting an entire workspace with `cargo clippy --all`
* New lint: [`naive_bytecount`]

## 0.0.153
* Update to *rustc 1.21.0-nightly (8c303ed87 2017-08-20)*
* New lint: [`use_self`]

## 0.0.152
* Update to *rustc 1.21.0-nightly (df511d554 2017-08-14)*

## 0.0.151
* Update to *rustc 1.21.0-nightly (13d94d5fa 2017-08-10)*

## 0.0.150
* Update to *rustc 1.21.0-nightly (215e0b10e 2017-08-08)*

## 0.0.148
* Update to *rustc 1.21.0-nightly (37c7d0ebb 2017-07-31)*
* New lints: [`unreadable_literal`], [`inconsistent_digit_grouping`], [`large_digit_groups`]

## 0.0.147
* Update to *rustc 1.21.0-nightly (aac223f4f 2017-07-30)*

## 0.0.146
* Update to *rustc 1.21.0-nightly (52a330969 2017-07-27)*
* Fixes false positives in `inline_always`
* Fixes false negatives in `panic_params`

## 0.0.145
* Update to *rustc 1.20.0-nightly (afe145d22 2017-07-23)*

## 0.0.144
* Update to *rustc 1.20.0-nightly (086eaa78e 2017-07-15)*

## 0.0.143
* Update to *rustc 1.20.0-nightly (d84693b93 2017-07-09)*
* Fix `cargo clippy` crashing on `dylib` projects
* Fix false positives around `nested_while_let` and `never_loop`

## 0.0.142
* Update to *rustc 1.20.0-nightly (067971139 2017-07-02)*

## 0.0.141
* Rewrite of the `doc_markdown` lint.
* Deprecated [`range_step_by_zero`]
* New lint: [`iterator_step_by_zero`]
* New lint: [`needless_borrowed_reference`]
* Update to *rustc 1.20.0-nightly (69c65d296 2017-06-28)*

## 0.0.140 - 2017-06-16
* Update to *rustc 1.19.0-nightly (258ae6dd9 2017-06-15)*

## 0.0.139 — 2017-06-10
* Update to *rustc 1.19.0-nightly (4bf5c99af 2017-06-10)*
* Fix bugs with for loop desugaring
* Check for [`AsRef`]/[`AsMut`] arguments in [`wrong_self_convention`]

## 0.0.138 — 2017-06-05
* Update to *rustc 1.19.0-nightly (0418fa9d3 2017-06-04)*

## 0.0.137 — 2017-06-05
* Update to *rustc 1.19.0-nightly (6684d176c 2017-06-03)*

## 0.0.136 — 2017—05—26
* Update to *rustc 1.19.0-nightly (557967766 2017-05-26)*

## 0.0.135 — 2017—05—24
* Update to *rustc 1.19.0-nightly (5b13bff52 2017-05-23)*

## 0.0.134 — 2017—05—19
* Update to *rustc 1.19.0-nightly (0ed1ec9f9 2017-05-18)*

## 0.0.133 — 2017—05—14
* Update to *rustc 1.19.0-nightly (826d8f385 2017-05-13)*

## 0.0.132 — 2017—05—05
* Fix various bugs and some ices

## 0.0.131 — 2017—05—04
* Update to *rustc 1.19.0-nightly (2d4ed8e0c 2017-05-03)*

## 0.0.130 — 2017—05—03
* Update to *rustc 1.19.0-nightly (6a5fc9eec 2017-05-02)*

## 0.0.129 — 2017-05-01
* Update to *rustc 1.19.0-nightly (06fb4d256 2017-04-30)*

## 0.0.128 — 2017-04-28
* Update to *rustc 1.18.0-nightly (94e884b63 2017-04-27)*

## 0.0.127 — 2017-04-27
* Update to *rustc 1.18.0-nightly (036983201 2017-04-26)*
* New lint: [`needless_continue`]

## 0.0.126 — 2017-04-24
* Update to *rustc 1.18.0-nightly (2bd4b5c6d 2017-04-23)*

## 0.0.125 — 2017-04-19
* Update to *rustc 1.18.0-nightly (9f2abadca 2017-04-18)*

## 0.0.124 — 2017-04-16
* Update to *rustc 1.18.0-nightly (d5cf1cb64 2017-04-15)*

## 0.0.123 — 2017-04-07
* Fix various false positives

## 0.0.122 — 2017-04-07
* Rustup to *rustc 1.18.0-nightly (91ae22a01 2017-04-05)*
* New lint: [`op_ref`]

## 0.0.121 — 2017-03-21
* Rustup to *rustc 1.17.0-nightly (134c4a0f0 2017-03-20)*

## 0.0.120 — 2017-03-17
* Rustup to *rustc 1.17.0-nightly (0aeb9c129 2017-03-15)*

## 0.0.119 — 2017-03-13
* Rustup to *rustc 1.17.0-nightly (824c9ebbd 2017-03-12)*

## 0.0.118 — 2017-03-05
* Rustup to *rustc 1.17.0-nightly (b1e31766d 2017-03-03)*

## 0.0.117 — 2017-03-01
* Rustup to *rustc 1.17.0-nightly (be760566c 2017-02-28)*

## 0.0.116 — 2017-02-28
* Fix `cargo clippy` on 64 bit windows systems

## 0.0.115 — 2017-02-27
* Rustup to *rustc 1.17.0-nightly (60a0edc6c 2017-02-26)*
* New lints: [`zero_ptr`], [`never_loop`], [`mut_from_ref`]

## 0.0.114 — 2017-02-08
* Rustup to *rustc 1.17.0-nightly (c49d10207 2017-02-07)*
* Tests are now ui tests (testing the exact output of rustc)

## 0.0.113 — 2017-02-04
* Rustup to *rustc 1.16.0-nightly (eedaa94e3 2017-02-02)*
* New lint: [`large_enum_variant`]
* `explicit_into_iter_loop` provides suggestions

## 0.0.112 — 2017-01-27
* Rustup to *rustc 1.16.0-nightly (df8debf6d 2017-01-25)*

## 0.0.111 — 2017-01-21
* Rustup to *rustc 1.16.0-nightly (a52da95ce 2017-01-20)*

## 0.0.110 — 2017-01-20
* Add badges and categories to `Cargo.toml`

## 0.0.109 — 2017-01-19
* Update to *rustc 1.16.0-nightly (c07a6ae77 2017-01-17)*

## 0.0.108 — 2017-01-12
* Update to *rustc 1.16.0-nightly (2782e8f8f 2017-01-12)*

## 0.0.107 — 2017-01-11
* Update regex dependency
* Fix FP when matching `&&mut` by `&ref`
* Reintroduce `for (_, x) in &mut hash_map` -> `for x in hash_map.values_mut()`
* New lints: [`unused_io_amount`], [`forget_ref`], [`short_circuit_statement`]

## 0.0.106 — 2017-01-04
* Fix FP introduced by rustup in [`wrong_self_convention`]

## 0.0.105 — 2017-01-04
* Update to *rustc 1.16.0-nightly (468227129 2017-01-03)*
* New lints: [`deref_addrof`], [`double_parens`], [`pub_enum_variant_names`]
* Fix suggestion in [`new_without_default`]
* FP fix in [`absurd_extreme_comparisons`]

## 0.0.104 — 2016-12-15
* Update to *rustc 1.15.0-nightly (8f02c429a 2016-12-15)*

## 0.0.103 — 2016-11-25
* Update to *rustc 1.15.0-nightly (d5814b03e 2016-11-23)*

## 0.0.102 — 2016-11-24
* Update to *rustc 1.15.0-nightly (3bf2be9ce 2016-11-22)*

## 0.0.101 — 2016-11-23
* Update to *rustc 1.15.0-nightly (7b3eeea22 2016-11-21)*
* New lint: [`string_extend_chars`]

## 0.0.100 — 2016-11-20
* Update to *rustc 1.15.0-nightly (ac635aa95 2016-11-18)*

## 0.0.99 — 2016-11-18
* Update to rustc 1.15.0-nightly (0ed951993 2016-11-14)
* New lint: [`get_unwrap`]

## 0.0.98 — 2016-11-08
* Fixes an issue due to a change in how cargo handles `--sysroot`, which broke `cargo clippy`

## 0.0.97 — 2016-11-03
* For convenience, `cargo clippy` defines a `cargo-clippy` feature. This was
  previously added for a short time under the name `clippy` but removed for
  compatibility.
* `cargo clippy --help` is more helping (and less helpful :smile:)
* Rustup to *rustc 1.14.0-nightly (5665bdf3e 2016-11-02)*
* New lints: [`if_let_redundant_pattern_matching`], [`partialeq_ne_impl`]

## 0.0.96 — 2016-10-22
* Rustup to *rustc 1.14.0-nightly (f09420685 2016-10-20)*
* New lint: [`iter_skip_next`]

## 0.0.95 — 2016-10-06
* Rustup to *rustc 1.14.0-nightly (3210fd5c2 2016-10-05)*

## 0.0.94 — 2016-10-04
* Fixes bustage on Windows due to forbidden directory name

## 0.0.93 — 2016-10-03
* Rustup to *rustc 1.14.0-nightly (144af3e97 2016-10-02)*
* `option_map_unwrap_or` and `option_map_unwrap_or_else` are now
  allowed by default.
* New lint: [`explicit_into_iter_loop`]

## 0.0.92 — 2016-09-30
* Rustup to *rustc 1.14.0-nightly (289f3a4ca 2016-09-29)*

## 0.0.91 — 2016-09-28
* Rustup to *rustc 1.13.0-nightly (d0623cf7b 2016-09-26)*

## 0.0.90 — 2016-09-09
* Rustup to *rustc 1.13.0-nightly (f1f40f850 2016-09-09)*

## 0.0.89 — 2016-09-06
* Rustup to *rustc 1.13.0-nightly (cbe4de78e 2016-09-05)*

## 0.0.88 — 2016-09-04
* Rustup to *rustc 1.13.0-nightly (70598e04f 2016-09-03)*
* The following lints are not new but were only usable through the `clippy`
  lint groups: [`filter_next`], `for_loop_over_option`,
  `for_loop_over_result` and [`match_overlapping_arm`]. You should now be
  able to `#[allow/deny]` them individually and they are available directly
  through `cargo clippy`.

## 0.0.87 — 2016-08-31
* Rustup to *rustc 1.13.0-nightly (eac41469d 2016-08-30)*
* New lints: [`builtin_type_shadow`]
* Fix FP in [`zero_prefixed_literal`] and `0b`/`0o`

## 0.0.86 — 2016-08-28
* Rustup to *rustc 1.13.0-nightly (a23064af5 2016-08-27)*
* New lints: [`missing_docs_in_private_items`], [`zero_prefixed_literal`]

## 0.0.85 — 2016-08-19
* Fix ICE with [`useless_attribute`]
* [`useless_attribute`] ignores `unused_imports` on `use` statements

## 0.0.84 — 2016-08-18
* Rustup to *rustc 1.13.0-nightly (aef6971ca 2016-08-17)*

## 0.0.83 — 2016-08-17
* Rustup to *rustc 1.12.0-nightly (1bf5fa326 2016-08-16)*
* New lints: [`print_with_newline`], [`useless_attribute`]

## 0.0.82 — 2016-08-17
* Rustup to *rustc 1.12.0-nightly (197be89f3 2016-08-15)*
* New lint: [`module_inception`]

## 0.0.81 — 2016-08-14
* Rustup to *rustc 1.12.0-nightly (1deb02ea6 2016-08-12)*
* New lints: [`eval_order_dependence`], [`mixed_case_hex_literals`], [`unseparated_literal_suffix`]
* False positive fix in [`too_many_arguments`]
* Addition of functionality to [`needless_borrow`]
* Suggestions for [`clone_on_copy`]
* Bug fix in [`wrong_self_convention`]
* Doc improvements

## 0.0.80 — 2016-07-31
* Rustup to *rustc 1.12.0-nightly (1225e122f 2016-07-30)*
* New lints: [`misrefactored_assign_op`], [`serde_api_misuse`]

## 0.0.79 — 2016-07-10
* Rustup to *rustc 1.12.0-nightly (f93aaf84c 2016-07-09)*
* Major suggestions refactoring

## 0.0.78 — 2016-07-02
* Rustup to *rustc 1.11.0-nightly (01411937f 2016-07-01)*
* New lints: [`wrong_transmute`], [`double_neg`], [`filter_map`]
* For compatibility, `cargo clippy` does not defines the `clippy` feature
  introduced in 0.0.76 anymore
* [`collapsible_if`] now considers `if let`

## 0.0.77 — 2016-06-21
* Rustup to *rustc 1.11.0-nightly (5522e678b 2016-06-20)*
* New lints: `stutter` and [`iter_nth`]

## 0.0.76 — 2016-06-10
* Rustup to *rustc 1.11.0-nightly (7d2f75a95 2016-06-09)*
* `cargo clippy` now automatically defines the `clippy` feature
* New lint: [`not_unsafe_ptr_arg_deref`]

## 0.0.75 — 2016-06-08
* Rustup to *rustc 1.11.0-nightly (763f9234b 2016-06-06)*

## 0.0.74 — 2016-06-07
* Fix bug with `cargo-clippy` JSON parsing
* Add the `CLIPPY_DISABLE_DOCS_LINKS` environment variable to deactivate the
  “for further information visit *lint-link*” message.

## 0.0.73 — 2016-06-05
* Fix false positives in [`useless_let_if_seq`]

## 0.0.72 — 2016-06-04
* Fix false positives in [`useless_let_if_seq`]

## 0.0.71 — 2016-05-31
* Rustup to *rustc 1.11.0-nightly (a967611d8 2016-05-30)*
* New lint: [`useless_let_if_seq`]

## 0.0.70 — 2016-05-28
* Rustup to *rustc 1.10.0-nightly (7bddce693 2016-05-27)*
* [`invalid_regex`] and [`trivial_regex`] can now warn on `RegexSet::new`,
  `RegexBuilder::new` and byte regexes

## 0.0.69 — 2016-05-20
* Rustup to *rustc 1.10.0-nightly (476fe6eef 2016-05-21)*
* [`used_underscore_binding`] has been made `Allow` temporarily

## 0.0.68 — 2016-05-17
* Rustup to *rustc 1.10.0-nightly (cd6a40017 2016-05-16)*
* New lint: [`unnecessary_operation`]

## 0.0.67 — 2016-05-12
* Rustup to *rustc 1.10.0-nightly (22ac88f1a 2016-05-11)*

## 0.0.66 — 2016-05-11
* New `cargo clippy` subcommand
* New lints: [`assign_op_pattern`], [`assign_ops`], [`needless_borrow`]

## 0.0.65 — 2016-05-08
* Rustup to *rustc 1.10.0-nightly (62e2b2fb7 2016-05-06)*
* New lints: [`float_arithmetic`], [`integer_arithmetic`]

## 0.0.64 — 2016-04-26
* Rustup to *rustc 1.10.0-nightly (645dd013a 2016-04-24)*
* New lints: [`temporary_cstring_as_ptr`], [`unsafe_removed_from_name`], and [`mem_forget`]

## 0.0.63 — 2016-04-08
* Rustup to *rustc 1.9.0-nightly (7979dd608 2016-04-07)*

## 0.0.62 — 2016-04-07
* Rustup to *rustc 1.9.0-nightly (bf5da36f1 2016-04-06)*

## 0.0.61 — 2016-04-03
* Rustup to *rustc 1.9.0-nightly (5ab11d72c 2016-04-02)*
* New lint: [`invalid_upcast_comparisons`]

## 0.0.60 — 2016-04-01
* Rustup to *rustc 1.9.0-nightly (e1195c24b 2016-03-31)*

## 0.0.59 — 2016-03-31
* Rustup to *rustc 1.9.0-nightly (30a3849f2 2016-03-30)*
* New lints: [`logic_bug`], [`nonminimal_bool`]
* Fixed: [`match_same_arms`] now ignores arms with guards
* Improved: [`useless_vec`] now warns on `for … in vec![…]`

## 0.0.58 — 2016-03-27
* Rustup to *rustc 1.9.0-nightly (d5a91e695 2016-03-26)*
* New lint: [`doc_markdown`]

## 0.0.57 — 2016-03-27
* Update to *rustc 1.9.0-nightly (a1e29daf1 2016-03-25)*
* Deprecated lints: [`str_to_string`], [`string_to_string`], [`unstable_as_slice`], [`unstable_as_mut_slice`]
* New lint: [`crosspointer_transmute`]

## 0.0.56 — 2016-03-23
* Update to *rustc 1.9.0-nightly (0dcc413e4 2016-03-22)*
* New lints: [`many_single_char_names`] and [`similar_names`]

## 0.0.55 — 2016-03-21
* Update to *rustc 1.9.0-nightly (02310fd31 2016-03-19)*

## 0.0.54 — 2016-03-16
* Update to *rustc 1.9.0-nightly (c66d2380a 2016-03-15)*

## 0.0.53 — 2016-03-15
* Add a [configuration file]

## ~~0.0.52~~

## 0.0.51 — 2016-03-13
* Add `str` to types considered by [`len_zero`]
* New lints: [`indexing_slicing`]

## 0.0.50 — 2016-03-11
* Update to *rustc 1.9.0-nightly (c9629d61c 2016-03-10)*

## 0.0.49 — 2016-03-09
* Update to *rustc 1.9.0-nightly (eabfc160f 2016-03-08)*
* New lints: [`overflow_check_conditional`], [`unused_label`], [`new_without_default`]

## 0.0.48 — 2016-03-07
* Fixed: ICE in [`needless_range_loop`] with globals

## 0.0.47 — 2016-03-07
* Update to *rustc 1.9.0-nightly (998a6720b 2016-03-07)*
* New lint: [`redundant_closure_call`]

[`AsMut`]: https://doc.rust-lang.org/std/convert/trait.AsMut.html
[`AsRef`]: https://doc.rust-lang.org/std/convert/trait.AsRef.html
[configuration file]: ./rust-clippy#configuration
[pull3665]: https://github.com/rust-lang/rust-clippy/pull/3665
[adding_lints]: https://github.com/rust-lang/rust-clippy/blob/master/doc/adding_lints.md

<!-- lint disable no-unused-definitions -->
<!-- begin autogenerated links to lint list -->
[`absurd_extreme_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#absurd_extreme_comparisons
[`almost_swapped`]: https://rust-lang.github.io/rust-clippy/master/index.html#almost_swapped
[`approx_constant`]: https://rust-lang.github.io/rust-clippy/master/index.html#approx_constant
[`as_conversions`]: https://rust-lang.github.io/rust-clippy/master/index.html#as_conversions
[`assertions_on_constants`]: https://rust-lang.github.io/rust-clippy/master/index.html#assertions_on_constants
[`assign_op_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#assign_op_pattern
[`assign_ops`]: https://rust-lang.github.io/rust-clippy/master/index.html#assign_ops
[`async_yields_async`]: https://rust-lang.github.io/rust-clippy/master/index.html#async_yields_async
[`await_holding_lock`]: https://rust-lang.github.io/rust-clippy/master/index.html#await_holding_lock
[`await_holding_refcell_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#await_holding_refcell_ref
[`bad_bit_mask`]: https://rust-lang.github.io/rust-clippy/master/index.html#bad_bit_mask
[`bind_instead_of_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#bind_instead_of_map
[`blacklisted_name`]: https://rust-lang.github.io/rust-clippy/master/index.html#blacklisted_name
[`blanket_clippy_restriction_lints`]: https://rust-lang.github.io/rust-clippy/master/index.html#blanket_clippy_restriction_lints
[`blocks_in_if_conditions`]: https://rust-lang.github.io/rust-clippy/master/index.html#blocks_in_if_conditions
[`bool_comparison`]: https://rust-lang.github.io/rust-clippy/master/index.html#bool_comparison
[`borrow_interior_mutable_const`]: https://rust-lang.github.io/rust-clippy/master/index.html#borrow_interior_mutable_const
[`borrowed_box`]: https://rust-lang.github.io/rust-clippy/master/index.html#borrowed_box
[`box_vec`]: https://rust-lang.github.io/rust-clippy/master/index.html#box_vec
[`boxed_local`]: https://rust-lang.github.io/rust-clippy/master/index.html#boxed_local
[`builtin_type_shadow`]: https://rust-lang.github.io/rust-clippy/master/index.html#builtin_type_shadow
[`cargo_common_metadata`]: https://rust-lang.github.io/rust-clippy/master/index.html#cargo_common_metadata
[`cast_lossless`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_lossless
[`cast_possible_truncation`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_possible_truncation
[`cast_possible_wrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_possible_wrap
[`cast_precision_loss`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_precision_loss
[`cast_ptr_alignment`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_ptr_alignment
[`cast_ref_to_mut`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_ref_to_mut
[`cast_sign_loss`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_sign_loss
[`char_lit_as_u8`]: https://rust-lang.github.io/rust-clippy/master/index.html#char_lit_as_u8
[`chars_last_cmp`]: https://rust-lang.github.io/rust-clippy/master/index.html#chars_last_cmp
[`chars_next_cmp`]: https://rust-lang.github.io/rust-clippy/master/index.html#chars_next_cmp
[`checked_conversions`]: https://rust-lang.github.io/rust-clippy/master/index.html#checked_conversions
[`clone_double_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#clone_double_ref
[`clone_on_copy`]: https://rust-lang.github.io/rust-clippy/master/index.html#clone_on_copy
[`clone_on_ref_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#clone_on_ref_ptr
[`cmp_nan`]: https://rust-lang.github.io/rust-clippy/master/index.html#cmp_nan
[`cmp_null`]: https://rust-lang.github.io/rust-clippy/master/index.html#cmp_null
[`cmp_owned`]: https://rust-lang.github.io/rust-clippy/master/index.html#cmp_owned
[`cognitive_complexity`]: https://rust-lang.github.io/rust-clippy/master/index.html#cognitive_complexity
[`collapsible_else_if`]: https://rust-lang.github.io/rust-clippy/master/index.html#collapsible_else_if
[`collapsible_if`]: https://rust-lang.github.io/rust-clippy/master/index.html#collapsible_if
[`collapsible_match`]: https://rust-lang.github.io/rust-clippy/master/index.html#collapsible_match
[`comparison_chain`]: https://rust-lang.github.io/rust-clippy/master/index.html#comparison_chain
[`comparison_to_empty`]: https://rust-lang.github.io/rust-clippy/master/index.html#comparison_to_empty
[`copy_iterator`]: https://rust-lang.github.io/rust-clippy/master/index.html#copy_iterator
[`create_dir`]: https://rust-lang.github.io/rust-clippy/master/index.html#create_dir
[`crosspointer_transmute`]: https://rust-lang.github.io/rust-clippy/master/index.html#crosspointer_transmute
[`dbg_macro`]: https://rust-lang.github.io/rust-clippy/master/index.html#dbg_macro
[`debug_assert_with_mut_call`]: https://rust-lang.github.io/rust-clippy/master/index.html#debug_assert_with_mut_call
[`decimal_literal_representation`]: https://rust-lang.github.io/rust-clippy/master/index.html#decimal_literal_representation
[`declare_interior_mutable_const`]: https://rust-lang.github.io/rust-clippy/master/index.html#declare_interior_mutable_const
[`default_trait_access`]: https://rust-lang.github.io/rust-clippy/master/index.html#default_trait_access
[`deprecated_cfg_attr`]: https://rust-lang.github.io/rust-clippy/master/index.html#deprecated_cfg_attr
[`deprecated_semver`]: https://rust-lang.github.io/rust-clippy/master/index.html#deprecated_semver
[`deref_addrof`]: https://rust-lang.github.io/rust-clippy/master/index.html#deref_addrof
[`derive_hash_xor_eq`]: https://rust-lang.github.io/rust-clippy/master/index.html#derive_hash_xor_eq
[`derive_ord_xor_partial_ord`]: https://rust-lang.github.io/rust-clippy/master/index.html#derive_ord_xor_partial_ord
[`disallowed_method`]: https://rust-lang.github.io/rust-clippy/master/index.html#disallowed_method
[`diverging_sub_expression`]: https://rust-lang.github.io/rust-clippy/master/index.html#diverging_sub_expression
[`doc_markdown`]: https://rust-lang.github.io/rust-clippy/master/index.html#doc_markdown
[`double_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#double_comparisons
[`double_must_use`]: https://rust-lang.github.io/rust-clippy/master/index.html#double_must_use
[`double_neg`]: https://rust-lang.github.io/rust-clippy/master/index.html#double_neg
[`double_parens`]: https://rust-lang.github.io/rust-clippy/master/index.html#double_parens
[`drop_bounds`]: https://rust-lang.github.io/rust-clippy/master/index.html#drop_bounds
[`drop_copy`]: https://rust-lang.github.io/rust-clippy/master/index.html#drop_copy
[`drop_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#drop_ref
[`duplicate_underscore_argument`]: https://rust-lang.github.io/rust-clippy/master/index.html#duplicate_underscore_argument
[`duration_subsec`]: https://rust-lang.github.io/rust-clippy/master/index.html#duration_subsec
[`else_if_without_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#else_if_without_else
[`empty_enum`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_enum
[`empty_line_after_outer_attr`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_line_after_outer_attr
[`empty_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_loop
[`enum_clike_unportable_variant`]: https://rust-lang.github.io/rust-clippy/master/index.html#enum_clike_unportable_variant
[`enum_glob_use`]: https://rust-lang.github.io/rust-clippy/master/index.html#enum_glob_use
[`enum_variant_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#enum_variant_names
[`eq_op`]: https://rust-lang.github.io/rust-clippy/master/index.html#eq_op
[`erasing_op`]: https://rust-lang.github.io/rust-clippy/master/index.html#erasing_op
[`eval_order_dependence`]: https://rust-lang.github.io/rust-clippy/master/index.html#eval_order_dependence
[`excessive_precision`]: https://rust-lang.github.io/rust-clippy/master/index.html#excessive_precision
[`exit`]: https://rust-lang.github.io/rust-clippy/master/index.html#exit
[`expect_fun_call`]: https://rust-lang.github.io/rust-clippy/master/index.html#expect_fun_call
[`expect_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#expect_used
[`expl_impl_clone_on_copy`]: https://rust-lang.github.io/rust-clippy/master/index.html#expl_impl_clone_on_copy
[`explicit_counter_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_counter_loop
[`explicit_deref_methods`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_deref_methods
[`explicit_into_iter_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_into_iter_loop
[`explicit_iter_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_iter_loop
[`explicit_write`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_write
[`extend_from_slice`]: https://rust-lang.github.io/rust-clippy/master/index.html#extend_from_slice
[`extra_unused_lifetimes`]: https://rust-lang.github.io/rust-clippy/master/index.html#extra_unused_lifetimes
[`fallible_impl_from`]: https://rust-lang.github.io/rust-clippy/master/index.html#fallible_impl_from
[`field_reassign_with_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#field_reassign_with_default
[`filetype_is_file`]: https://rust-lang.github.io/rust-clippy/master/index.html#filetype_is_file
[`filter_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#filter_map
[`filter_map_next`]: https://rust-lang.github.io/rust-clippy/master/index.html#filter_map_next
[`filter_next`]: https://rust-lang.github.io/rust-clippy/master/index.html#filter_next
[`find_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#find_map
[`flat_map_identity`]: https://rust-lang.github.io/rust-clippy/master/index.html#flat_map_identity
[`float_arithmetic`]: https://rust-lang.github.io/rust-clippy/master/index.html#float_arithmetic
[`float_cmp`]: https://rust-lang.github.io/rust-clippy/master/index.html#float_cmp
[`float_cmp_const`]: https://rust-lang.github.io/rust-clippy/master/index.html#float_cmp_const
[`float_equality_without_abs`]: https://rust-lang.github.io/rust-clippy/master/index.html#float_equality_without_abs
[`fn_address_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_address_comparisons
[`fn_params_excessive_bools`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_params_excessive_bools
[`fn_to_numeric_cast`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_to_numeric_cast
[`fn_to_numeric_cast_with_truncation`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_to_numeric_cast_with_truncation
[`for_kv_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#for_kv_map
[`for_loops_over_fallibles`]: https://rust-lang.github.io/rust-clippy/master/index.html#for_loops_over_fallibles
[`forget_copy`]: https://rust-lang.github.io/rust-clippy/master/index.html#forget_copy
[`forget_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#forget_ref
[`from_iter_instead_of_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#from_iter_instead_of_collect
[`from_over_into`]: https://rust-lang.github.io/rust-clippy/master/index.html#from_over_into
[`future_not_send`]: https://rust-lang.github.io/rust-clippy/master/index.html#future_not_send
[`get_last_with_len`]: https://rust-lang.github.io/rust-clippy/master/index.html#get_last_with_len
[`get_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#get_unwrap
[`identity_op`]: https://rust-lang.github.io/rust-clippy/master/index.html#identity_op
[`if_let_mutex`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_let_mutex
[`if_let_redundant_pattern_matching`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_let_redundant_pattern_matching
[`if_let_some_result`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_let_some_result
[`if_not_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_not_else
[`if_same_then_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_same_then_else
[`ifs_same_cond`]: https://rust-lang.github.io/rust-clippy/master/index.html#ifs_same_cond
[`implicit_hasher`]: https://rust-lang.github.io/rust-clippy/master/index.html#implicit_hasher
[`implicit_return`]: https://rust-lang.github.io/rust-clippy/master/index.html#implicit_return
[`implicit_saturating_sub`]: https://rust-lang.github.io/rust-clippy/master/index.html#implicit_saturating_sub
[`imprecise_flops`]: https://rust-lang.github.io/rust-clippy/master/index.html#imprecise_flops
[`inconsistent_digit_grouping`]: https://rust-lang.github.io/rust-clippy/master/index.html#inconsistent_digit_grouping
[`indexing_slicing`]: https://rust-lang.github.io/rust-clippy/master/index.html#indexing_slicing
[`ineffective_bit_mask`]: https://rust-lang.github.io/rust-clippy/master/index.html#ineffective_bit_mask
[`inefficient_to_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#inefficient_to_string
[`infallible_destructuring_match`]: https://rust-lang.github.io/rust-clippy/master/index.html#infallible_destructuring_match
[`infinite_iter`]: https://rust-lang.github.io/rust-clippy/master/index.html#infinite_iter
[`inherent_to_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#inherent_to_string
[`inherent_to_string_shadow_display`]: https://rust-lang.github.io/rust-clippy/master/index.html#inherent_to_string_shadow_display
[`inline_always`]: https://rust-lang.github.io/rust-clippy/master/index.html#inline_always
[`inline_asm_x86_att_syntax`]: https://rust-lang.github.io/rust-clippy/master/index.html#inline_asm_x86_att_syntax
[`inline_asm_x86_intel_syntax`]: https://rust-lang.github.io/rust-clippy/master/index.html#inline_asm_x86_intel_syntax
[`inline_fn_without_body`]: https://rust-lang.github.io/rust-clippy/master/index.html#inline_fn_without_body
[`int_plus_one`]: https://rust-lang.github.io/rust-clippy/master/index.html#int_plus_one
[`integer_arithmetic`]: https://rust-lang.github.io/rust-clippy/master/index.html#integer_arithmetic
[`integer_division`]: https://rust-lang.github.io/rust-clippy/master/index.html#integer_division
[`into_iter_on_array`]: https://rust-lang.github.io/rust-clippy/master/index.html#into_iter_on_array
[`into_iter_on_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#into_iter_on_ref
[`invalid_atomic_ordering`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_atomic_ordering
[`invalid_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_ref
[`invalid_regex`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_regex
[`invalid_upcast_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_upcast_comparisons
[`invisible_characters`]: https://rust-lang.github.io/rust-clippy/master/index.html#invisible_characters
[`items_after_statements`]: https://rust-lang.github.io/rust-clippy/master/index.html#items_after_statements
[`iter_cloned_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_cloned_collect
[`iter_next_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_next_loop
[`iter_next_slice`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_next_slice
[`iter_nth`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_nth
[`iter_nth_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_nth_zero
[`iter_skip_next`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_skip_next
[`iterator_step_by_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#iterator_step_by_zero
[`just_underscores_and_digits`]: https://rust-lang.github.io/rust-clippy/master/index.html#just_underscores_and_digits
[`large_const_arrays`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_const_arrays
[`large_digit_groups`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_digit_groups
[`large_enum_variant`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_enum_variant
[`large_stack_arrays`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_stack_arrays
[`large_types_passed_by_value`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_types_passed_by_value
[`len_without_is_empty`]: https://rust-lang.github.io/rust-clippy/master/index.html#len_without_is_empty
[`len_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#len_zero
[`let_and_return`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_and_return
[`let_underscore_drop`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_underscore_drop
[`let_underscore_lock`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_underscore_lock
[`let_underscore_must_use`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_underscore_must_use
[`let_unit_value`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_unit_value
[`linkedlist`]: https://rust-lang.github.io/rust-clippy/master/index.html#linkedlist
[`logic_bug`]: https://rust-lang.github.io/rust-clippy/master/index.html#logic_bug
[`lossy_float_literal`]: https://rust-lang.github.io/rust-clippy/master/index.html#lossy_float_literal
[`macro_use_imports`]: https://rust-lang.github.io/rust-clippy/master/index.html#macro_use_imports
[`main_recursion`]: https://rust-lang.github.io/rust-clippy/master/index.html#main_recursion
[`manual_async_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_async_fn
[`manual_memcpy`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_memcpy
[`manual_non_exhaustive`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_non_exhaustive
[`manual_ok_or`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_ok_or
[`manual_range_contains`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_range_contains
[`manual_saturating_arithmetic`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_saturating_arithmetic
[`manual_strip`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_strip
[`manual_swap`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_swap
[`manual_unwrap_or`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_unwrap_or
[`many_single_char_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#many_single_char_names
[`map_clone`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_clone
[`map_collect_result_unit`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_collect_result_unit
[`map_entry`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_entry
[`map_err_ignore`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_err_ignore
[`map_flatten`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_flatten
[`map_identity`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_identity
[`map_unwrap_or`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_unwrap_or
[`match_as_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_as_ref
[`match_bool`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_bool
[`match_like_matches_macro`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_like_matches_macro
[`match_on_vec_items`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_on_vec_items
[`match_overlapping_arm`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_overlapping_arm
[`match_ref_pats`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_ref_pats
[`match_same_arms`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_same_arms
[`match_single_binding`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_single_binding
[`match_wild_err_arm`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_wild_err_arm
[`match_wildcard_for_single_variants`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_wildcard_for_single_variants
[`maybe_infinite_iter`]: https://rust-lang.github.io/rust-clippy/master/index.html#maybe_infinite_iter
[`mem_discriminant_non_enum`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_discriminant_non_enum
[`mem_forget`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_forget
[`mem_replace_option_with_none`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_replace_option_with_none
[`mem_replace_with_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_replace_with_default
[`mem_replace_with_uninit`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_replace_with_uninit
[`min_max`]: https://rust-lang.github.io/rust-clippy/master/index.html#min_max
[`misaligned_transmute`]: https://rust-lang.github.io/rust-clippy/master/index.html#misaligned_transmute
[`mismatched_target_os`]: https://rust-lang.github.io/rust-clippy/master/index.html#mismatched_target_os
[`misrefactored_assign_op`]: https://rust-lang.github.io/rust-clippy/master/index.html#misrefactored_assign_op
[`missing_const_for_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_const_for_fn
[`missing_docs_in_private_items`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_docs_in_private_items
[`missing_errors_doc`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_errors_doc
[`missing_inline_in_public_items`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_inline_in_public_items
[`missing_safety_doc`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_safety_doc
[`mistyped_literal_suffixes`]: https://rust-lang.github.io/rust-clippy/master/index.html#mistyped_literal_suffixes
[`mixed_case_hex_literals`]: https://rust-lang.github.io/rust-clippy/master/index.html#mixed_case_hex_literals
[`module_inception`]: https://rust-lang.github.io/rust-clippy/master/index.html#module_inception
[`module_name_repetitions`]: https://rust-lang.github.io/rust-clippy/master/index.html#module_name_repetitions
[`modulo_arithmetic`]: https://rust-lang.github.io/rust-clippy/master/index.html#modulo_arithmetic
[`modulo_one`]: https://rust-lang.github.io/rust-clippy/master/index.html#modulo_one
[`multiple_crate_versions`]: https://rust-lang.github.io/rust-clippy/master/index.html#multiple_crate_versions
[`multiple_inherent_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#multiple_inherent_impl
[`must_use_candidate`]: https://rust-lang.github.io/rust-clippy/master/index.html#must_use_candidate
[`must_use_unit`]: https://rust-lang.github.io/rust-clippy/master/index.html#must_use_unit
[`mut_from_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#mut_from_ref
[`mut_mut`]: https://rust-lang.github.io/rust-clippy/master/index.html#mut_mut
[`mut_mutex_lock`]: https://rust-lang.github.io/rust-clippy/master/index.html#mut_mutex_lock
[`mut_range_bound`]: https://rust-lang.github.io/rust-clippy/master/index.html#mut_range_bound
[`mutable_key_type`]: https://rust-lang.github.io/rust-clippy/master/index.html#mutable_key_type
[`mutex_atomic`]: https://rust-lang.github.io/rust-clippy/master/index.html#mutex_atomic
[`mutex_integer`]: https://rust-lang.github.io/rust-clippy/master/index.html#mutex_integer
[`naive_bytecount`]: https://rust-lang.github.io/rust-clippy/master/index.html#naive_bytecount
[`needless_arbitrary_self_type`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_arbitrary_self_type
[`needless_bool`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_bool
[`needless_borrow`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_borrow
[`needless_borrowed_reference`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_borrowed_reference
[`needless_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_collect
[`needless_continue`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_continue
[`needless_doctest_main`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_doctest_main
[`needless_lifetimes`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_lifetimes
[`needless_pass_by_value`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_pass_by_value
[`needless_question_mark`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_question_mark
[`needless_range_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_range_loop
[`needless_return`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_return
[`needless_update`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_update
[`neg_cmp_op_on_partial_ord`]: https://rust-lang.github.io/rust-clippy/master/index.html#neg_cmp_op_on_partial_ord
[`neg_multiply`]: https://rust-lang.github.io/rust-clippy/master/index.html#neg_multiply
[`never_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#never_loop
[`new_ret_no_self`]: https://rust-lang.github.io/rust-clippy/master/index.html#new_ret_no_self
[`new_without_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#new_without_default
[`no_effect`]: https://rust-lang.github.io/rust-clippy/master/index.html#no_effect
[`non_ascii_literal`]: https://rust-lang.github.io/rust-clippy/master/index.html#non_ascii_literal
[`nonminimal_bool`]: https://rust-lang.github.io/rust-clippy/master/index.html#nonminimal_bool
[`nonsensical_open_options`]: https://rust-lang.github.io/rust-clippy/master/index.html#nonsensical_open_options
[`not_unsafe_ptr_arg_deref`]: https://rust-lang.github.io/rust-clippy/master/index.html#not_unsafe_ptr_arg_deref
[`ok_expect`]: https://rust-lang.github.io/rust-clippy/master/index.html#ok_expect
[`op_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#op_ref
[`option_as_ref_deref`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_as_ref_deref
[`option_env_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_env_unwrap
[`option_if_let_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_if_let_else
[`option_map_or_none`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_map_or_none
[`option_map_unit_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_map_unit_fn
[`option_option`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_option
[`or_fun_call`]: https://rust-lang.github.io/rust-clippy/master/index.html#or_fun_call
[`out_of_bounds_indexing`]: https://rust-lang.github.io/rust-clippy/master/index.html#out_of_bounds_indexing
[`overflow_check_conditional`]: https://rust-lang.github.io/rust-clippy/master/index.html#overflow_check_conditional
[`panic`]: https://rust-lang.github.io/rust-clippy/master/index.html#panic
[`panic_in_result_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#panic_in_result_fn
[`panic_params`]: https://rust-lang.github.io/rust-clippy/master/index.html#panic_params
[`panicking_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#panicking_unwrap
[`partialeq_ne_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#partialeq_ne_impl
[`path_buf_push_overwrite`]: https://rust-lang.github.io/rust-clippy/master/index.html#path_buf_push_overwrite
[`pattern_type_mismatch`]: https://rust-lang.github.io/rust-clippy/master/index.html#pattern_type_mismatch
[`possible_missing_comma`]: https://rust-lang.github.io/rust-clippy/master/index.html#possible_missing_comma
[`precedence`]: https://rust-lang.github.io/rust-clippy/master/index.html#precedence
[`print_literal`]: https://rust-lang.github.io/rust-clippy/master/index.html#print_literal
[`print_stderr`]: https://rust-lang.github.io/rust-clippy/master/index.html#print_stderr
[`print_stdout`]: https://rust-lang.github.io/rust-clippy/master/index.html#print_stdout
[`print_with_newline`]: https://rust-lang.github.io/rust-clippy/master/index.html#print_with_newline
[`println_empty_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#println_empty_string
[`ptr_arg`]: https://rust-lang.github.io/rust-clippy/master/index.html#ptr_arg
[`ptr_as_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#ptr_as_ptr
[`ptr_eq`]: https://rust-lang.github.io/rust-clippy/master/index.html#ptr_eq
[`ptr_offset_with_cast`]: https://rust-lang.github.io/rust-clippy/master/index.html#ptr_offset_with_cast
[`pub_enum_variant_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#pub_enum_variant_names
[`question_mark`]: https://rust-lang.github.io/rust-clippy/master/index.html#question_mark
[`range_minus_one`]: https://rust-lang.github.io/rust-clippy/master/index.html#range_minus_one
[`range_plus_one`]: https://rust-lang.github.io/rust-clippy/master/index.html#range_plus_one
[`range_step_by_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#range_step_by_zero
[`range_zip_with_len`]: https://rust-lang.github.io/rust-clippy/master/index.html#range_zip_with_len
[`rc_buffer`]: https://rust-lang.github.io/rust-clippy/master/index.html#rc_buffer
[`redundant_allocation`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_allocation
[`redundant_clone`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_clone
[`redundant_closure`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_closure
[`redundant_closure_call`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_closure_call
[`redundant_closure_for_method_calls`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_closure_for_method_calls
[`redundant_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_else
[`redundant_field_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_field_names
[`redundant_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_pattern
[`redundant_pattern_matching`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_pattern_matching
[`redundant_pub_crate`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_pub_crate
[`redundant_static_lifetimes`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_static_lifetimes
[`ref_in_deref`]: https://rust-lang.github.io/rust-clippy/master/index.html#ref_in_deref
[`ref_option_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#ref_option_ref
[`regex_macro`]: https://rust-lang.github.io/rust-clippy/master/index.html#regex_macro
[`repeat_once`]: https://rust-lang.github.io/rust-clippy/master/index.html#repeat_once
[`replace_consts`]: https://rust-lang.github.io/rust-clippy/master/index.html#replace_consts
[`rest_pat_in_fully_bound_structs`]: https://rust-lang.github.io/rust-clippy/master/index.html#rest_pat_in_fully_bound_structs
[`result_map_or_into_option`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_map_or_into_option
[`result_map_unit_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_map_unit_fn
[`result_unit_err`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_unit_err
[`reversed_empty_ranges`]: https://rust-lang.github.io/rust-clippy/master/index.html#reversed_empty_ranges
[`same_functions_in_if_condition`]: https://rust-lang.github.io/rust-clippy/master/index.html#same_functions_in_if_condition
[`same_item_push`]: https://rust-lang.github.io/rust-clippy/master/index.html#same_item_push
[`search_is_some`]: https://rust-lang.github.io/rust-clippy/master/index.html#search_is_some
[`self_assignment`]: https://rust-lang.github.io/rust-clippy/master/index.html#self_assignment
[`serde_api_misuse`]: https://rust-lang.github.io/rust-clippy/master/index.html#serde_api_misuse
[`shadow_reuse`]: https://rust-lang.github.io/rust-clippy/master/index.html#shadow_reuse
[`shadow_same`]: https://rust-lang.github.io/rust-clippy/master/index.html#shadow_same
[`shadow_unrelated`]: https://rust-lang.github.io/rust-clippy/master/index.html#shadow_unrelated
[`short_circuit_statement`]: https://rust-lang.github.io/rust-clippy/master/index.html#short_circuit_statement
[`should_assert_eq`]: https://rust-lang.github.io/rust-clippy/master/index.html#should_assert_eq
[`should_implement_trait`]: https://rust-lang.github.io/rust-clippy/master/index.html#should_implement_trait
[`similar_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#similar_names
[`single_char_add_str`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_char_add_str
[`single_char_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_char_pattern
[`single_component_path_imports`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_component_path_imports
[`single_element_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_element_loop
[`single_match`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_match
[`single_match_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_match_else
[`size_of_in_element_count`]: https://rust-lang.github.io/rust-clippy/master/index.html#size_of_in_element_count
[`skip_while_next`]: https://rust-lang.github.io/rust-clippy/master/index.html#skip_while_next
[`slow_vector_initialization`]: https://rust-lang.github.io/rust-clippy/master/index.html#slow_vector_initialization
[`stable_sort_primitive`]: https://rust-lang.github.io/rust-clippy/master/index.html#stable_sort_primitive
[`str_to_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#str_to_string
[`string_add`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_add
[`string_add_assign`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_add_assign
[`string_extend_chars`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_extend_chars
[`string_from_utf8_as_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_from_utf8_as_bytes
[`string_lit_as_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_lit_as_bytes
[`string_to_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_to_string
[`struct_excessive_bools`]: https://rust-lang.github.io/rust-clippy/master/index.html#struct_excessive_bools
[`suboptimal_flops`]: https://rust-lang.github.io/rust-clippy/master/index.html#suboptimal_flops
[`suspicious_arithmetic_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_arithmetic_impl
[`suspicious_assignment_formatting`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_assignment_formatting
[`suspicious_else_formatting`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_else_formatting
[`suspicious_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_map
[`suspicious_op_assign_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_op_assign_impl
[`suspicious_operation_groupings`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_operation_groupings
[`suspicious_unary_op_formatting`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_unary_op_formatting
[`tabs_in_doc_comments`]: https://rust-lang.github.io/rust-clippy/master/index.html#tabs_in_doc_comments
[`temporary_assignment`]: https://rust-lang.github.io/rust-clippy/master/index.html#temporary_assignment
[`temporary_cstring_as_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#temporary_cstring_as_ptr
[`to_digit_is_some`]: https://rust-lang.github.io/rust-clippy/master/index.html#to_digit_is_some
[`to_string_in_display`]: https://rust-lang.github.io/rust-clippy/master/index.html#to_string_in_display
[`todo`]: https://rust-lang.github.io/rust-clippy/master/index.html#todo
[`too_many_arguments`]: https://rust-lang.github.io/rust-clippy/master/index.html#too_many_arguments
[`too_many_lines`]: https://rust-lang.github.io/rust-clippy/master/index.html#too_many_lines
[`toplevel_ref_arg`]: https://rust-lang.github.io/rust-clippy/master/index.html#toplevel_ref_arg
[`trait_duplication_in_bounds`]: https://rust-lang.github.io/rust-clippy/master/index.html#trait_duplication_in_bounds
[`transmute_bytes_to_str`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_bytes_to_str
[`transmute_float_to_int`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_float_to_int
[`transmute_int_to_bool`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_int_to_bool
[`transmute_int_to_char`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_int_to_char
[`transmute_int_to_float`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_int_to_float
[`transmute_ptr_to_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_ptr_to_ptr
[`transmute_ptr_to_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_ptr_to_ref
[`transmutes_expressible_as_ptr_casts`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmutes_expressible_as_ptr_casts
[`transmuting_null`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmuting_null
[`trivial_regex`]: https://rust-lang.github.io/rust-clippy/master/index.html#trivial_regex
[`trivially_copy_pass_by_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#trivially_copy_pass_by_ref
[`try_err`]: https://rust-lang.github.io/rust-clippy/master/index.html#try_err
[`type_complexity`]: https://rust-lang.github.io/rust-clippy/master/index.html#type_complexity
[`type_repetition_in_bounds`]: https://rust-lang.github.io/rust-clippy/master/index.html#type_repetition_in_bounds
[`undropped_manually_drops`]: https://rust-lang.github.io/rust-clippy/master/index.html#undropped_manually_drops
[`unicode_not_nfc`]: https://rust-lang.github.io/rust-clippy/master/index.html#unicode_not_nfc
[`unimplemented`]: https://rust-lang.github.io/rust-clippy/master/index.html#unimplemented
[`uninit_assumed_init`]: https://rust-lang.github.io/rust-clippy/master/index.html#uninit_assumed_init
[`unit_arg`]: https://rust-lang.github.io/rust-clippy/master/index.html#unit_arg
[`unit_cmp`]: https://rust-lang.github.io/rust-clippy/master/index.html#unit_cmp
[`unit_return_expecting_ord`]: https://rust-lang.github.io/rust-clippy/master/index.html#unit_return_expecting_ord
[`unknown_clippy_lints`]: https://rust-lang.github.io/rust-clippy/master/index.html#unknown_clippy_lints
[`unnecessary_cast`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_cast
[`unnecessary_filter_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_filter_map
[`unnecessary_fold`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_fold
[`unnecessary_lazy_evaluations`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_lazy_evaluations
[`unnecessary_mut_passed`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_mut_passed
[`unnecessary_operation`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_operation
[`unnecessary_sort_by`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_sort_by
[`unnecessary_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_unwrap
[`unnecessary_wraps`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_wraps
[`unneeded_field_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#unneeded_field_pattern
[`unneeded_wildcard_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#unneeded_wildcard_pattern
[`unnested_or_patterns`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnested_or_patterns
[`unreachable`]: https://rust-lang.github.io/rust-clippy/master/index.html#unreachable
[`unreadable_literal`]: https://rust-lang.github.io/rust-clippy/master/index.html#unreadable_literal
[`unsafe_derive_deserialize`]: https://rust-lang.github.io/rust-clippy/master/index.html#unsafe_derive_deserialize
[`unsafe_removed_from_name`]: https://rust-lang.github.io/rust-clippy/master/index.html#unsafe_removed_from_name
[`unsafe_vector_initialization`]: https://rust-lang.github.io/rust-clippy/master/index.html#unsafe_vector_initialization
[`unseparated_literal_suffix`]: https://rust-lang.github.io/rust-clippy/master/index.html#unseparated_literal_suffix
[`unsound_collection_transmute`]: https://rust-lang.github.io/rust-clippy/master/index.html#unsound_collection_transmute
[`unstable_as_mut_slice`]: https://rust-lang.github.io/rust-clippy/master/index.html#unstable_as_mut_slice
[`unstable_as_slice`]: https://rust-lang.github.io/rust-clippy/master/index.html#unstable_as_slice
[`unused_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_collect
[`unused_io_amount`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_io_amount
[`unused_label`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_label
[`unused_self`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_self
[`unused_unit`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_unit
[`unusual_byte_groupings`]: https://rust-lang.github.io/rust-clippy/master/index.html#unusual_byte_groupings
[`unwrap_in_result`]: https://rust-lang.github.io/rust-clippy/master/index.html#unwrap_in_result
[`unwrap_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#unwrap_used
[`use_debug`]: https://rust-lang.github.io/rust-clippy/master/index.html#use_debug
[`use_self`]: https://rust-lang.github.io/rust-clippy/master/index.html#use_self
[`used_underscore_binding`]: https://rust-lang.github.io/rust-clippy/master/index.html#used_underscore_binding
[`useless_asref`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_asref
[`useless_attribute`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_attribute
[`useless_conversion`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_conversion
[`useless_format`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_format
[`useless_let_if_seq`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_let_if_seq
[`useless_transmute`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_transmute
[`useless_vec`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_vec
[`vec_box`]: https://rust-lang.github.io/rust-clippy/master/index.html#vec_box
[`vec_init_then_push`]: https://rust-lang.github.io/rust-clippy/master/index.html#vec_init_then_push
[`vec_resize_to_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#vec_resize_to_zero
[`verbose_bit_mask`]: https://rust-lang.github.io/rust-clippy/master/index.html#verbose_bit_mask
[`verbose_file_reads`]: https://rust-lang.github.io/rust-clippy/master/index.html#verbose_file_reads
[`vtable_address_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#vtable_address_comparisons
[`while_immutable_condition`]: https://rust-lang.github.io/rust-clippy/master/index.html#while_immutable_condition
[`while_let_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#while_let_loop
[`while_let_on_iterator`]: https://rust-lang.github.io/rust-clippy/master/index.html#while_let_on_iterator
[`wildcard_dependencies`]: https://rust-lang.github.io/rust-clippy/master/index.html#wildcard_dependencies
[`wildcard_enum_match_arm`]: https://rust-lang.github.io/rust-clippy/master/index.html#wildcard_enum_match_arm
[`wildcard_imports`]: https://rust-lang.github.io/rust-clippy/master/index.html#wildcard_imports
[`wildcard_in_or_patterns`]: https://rust-lang.github.io/rust-clippy/master/index.html#wildcard_in_or_patterns
[`write_literal`]: https://rust-lang.github.io/rust-clippy/master/index.html#write_literal
[`write_with_newline`]: https://rust-lang.github.io/rust-clippy/master/index.html#write_with_newline
[`writeln_empty_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#writeln_empty_string
[`wrong_pub_self_convention`]: https://rust-lang.github.io/rust-clippy/master/index.html#wrong_pub_self_convention
[`wrong_self_convention`]: https://rust-lang.github.io/rust-clippy/master/index.html#wrong_self_convention
[`wrong_transmute`]: https://rust-lang.github.io/rust-clippy/master/index.html#wrong_transmute
[`zero_divided_by_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#zero_divided_by_zero
[`zero_prefixed_literal`]: https://rust-lang.github.io/rust-clippy/master/index.html#zero_prefixed_literal
[`zero_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#zero_ptr
[`zero_sized_map_values`]: https://rust-lang.github.io/rust-clippy/master/index.html#zero_sized_map_values
[`zst_offset`]: https://rust-lang.github.io/rust-clippy/master/index.html#zst_offset
<!-- end autogenerated links to lint list -->
