# Changelog

All notable changes to this project will be documented in this file.
See [Changelog Update](book/src/development/infrastructure/changelog_update.md) if you want to update this
document.

## Unreleased / Beta / In Rust Nightly

[1e5237f4...master](https://github.com/rust-lang/rust-clippy/compare/1e5237f4...master)

## Rust 1.87

Current stable, released 2025-05-15

[View all 127 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2025-02-06T14%3A54%3A28Z..2025-03-20T20%3A07%3A53Z+base%3Amaster)

### New Lints

* Added [`doc_comment_double_space_linebreaks`] to `pedantic` [#12876](https://github.com/rust-lang/rust-clippy/pull/12876)
* Added [`manual_midpoint`] to `pedantic` [#13851](https://github.com/rust-lang/rust-clippy/pull/13851)
* Added [`io_other_error`] to `style` [#14022](https://github.com/rust-lang/rust-clippy/pull/14022)
* Added [`owned_cow`] to `pedantic` [#13948](https://github.com/rust-lang/rust-clippy/pull/13948)
* Added [`manual_contains`] to `perf` [#13817](https://github.com/rust-lang/rust-clippy/pull/13817)
* Added [`unnecessary_debug_formatting`] to `pedantic` [#13893](https://github.com/rust-lang/rust-clippy/pull/13893)
* Added [`elidable_lifetime_names`] to `pedantic` [#13960](https://github.com/rust-lang/rust-clippy/pull/13960)
* Added [`mem_replace_option_with_some`] to `style` [#14197](https://github.com/rust-lang/rust-clippy/pull/14197)
* Added [`unbuffered_bytes`] to `perf` [#14089](https://github.com/rust-lang/rust-clippy/pull/14089)
* Added [`single_option_map`] to `nursery` [#14033](https://github.com/rust-lang/rust-clippy/pull/14033)

### Moves and Deprecations

* Moved [`comparison_chain`] to `pedantic` (from `style`)
  [#14219](https://github.com/rust-lang/rust-clippy/pull/14219)
* Moved [`manual_ok_or`] to `style` (from `pedantic`)
  [#14027](https://github.com/rust-lang/rust-clippy/pull/14027)
* Deprecated [`option_map_or_err_ok`] in favor of [`manual_ok_or`]
  [#14027](https://github.com/rust-lang/rust-clippy/pull/14027)

### Enhancements

* Add `allow_expect_in_consts` and `allow_unwrap_in_consts` configuration options to [`unwrap_used`], [`expect_used`]
  [#14200](https://github.com/rust-lang/rust-clippy/pull/14200)
* Add `check-incompatible-msrv-in-tests` configuration option to [`incompatible_msrv`]
  [#14279](https://github.com/rust-lang/rust-clippy/pull/14279)
* [`len_zero`] now also triggers if deref target implements `is_empty()`
  [#13871](https://github.com/rust-lang/rust-clippy/pull/13871)
* [`ptr_eq`] now handles more cases, including `!=` in addition to `==`
  [#14339](https://github.com/rust-lang/rust-clippy/pull/14339)
* [`struct_field_names`] now also checks private fields of public structs
  [#14076](https://github.com/rust-lang/rust-clippy/pull/14076)
* [`needless_pass_by_value`] suggests using a reference on the innermost `Option` content
  [#14392](https://github.com/rust-lang/rust-clippy/pull/14392)
* [`obfuscated_if_else`] now supports `then().unwrap_or_else()` and `then_some().unwrap_or_else()`
  [#14165](https://github.com/rust-lang/rust-clippy/pull/14165)
* Format macros: all format-handling lints now validate `todo!` and `unimplemented!` macros
  [#14266](https://github.com/rust-lang/rust-clippy/pull/14266)
* [`disallowed_methods`] now supports replacements
  [#13669](https://github.com/rust-lang/rust-clippy/pull/13669)
* Added MSRV checks for several lints:
  * [`question_mark`] [#14436](https://github.com/rust-lang/rust-clippy/pull/14436)
  * [`repeat_vec_with_capacity`] [#14126](https://github.com/rust-lang/rust-clippy/pull/14126)
  * [`manual_flatten`] [#14086](https://github.com/rust-lang/rust-clippy/pull/14086)
  * [`lines_filter_map_ok`] [#14130](https://github.com/rust-lang/rust-clippy/pull/14130)

### False Positive Fixes

* [`missing_const_for_fn`] no longer triggers on unstable const traits [#14294](https://github.com/rust-lang/rust-clippy/pull/14294)
* [`unnecessary_to_owned`] now avoids suggesting to call `iter()` on a temporary object [#14243](https://github.com/rust-lang/rust-clippy/pull/14243)
* [`unnecessary_debug_formatting`] no longer triggers in tests [#14347](https://github.com/rust-lang/rust-clippy/pull/14347)
* [`option_if_let_else`] now handles cases when value is partially moved [#14209](https://github.com/rust-lang/rust-clippy/pull/14209)
* [`blocks_in_conditions`] no longer triggers when the condition contains a `return` [#14338](https://github.com/rust-lang/rust-clippy/pull/14338)
* [`undocumented_unsafe_blocks`] no longer triggers on trait/impl items [#13888](https://github.com/rust-lang/rust-clippy/pull/13888)
* [`manual_slice_fill`] no longer triggers due to missing index checks [#14193](https://github.com/rust-lang/rust-clippy/pull/14193)
* [`useless_asref`] no longer suggests using `.clone()` if the target type doesn't implement `Clone` [#14174](https://github.com/rust-lang/rust-clippy/pull/14174)
* [`unnecessary_safety_comment`] no longer triggers on desugared assign [#14371](https://github.com/rust-lang/rust-clippy/pull/14371)
* [`unnecessary_map_or`] no longer consumes the comparison value if it does not implement `Copy` [#14207](https://github.com/rust-lang/rust-clippy/pull/14207)
* [`let_and_return`] no longer triggers involving short-lived block temporary variables [#14180](https://github.com/rust-lang/rust-clippy/pull/14180)
* [`manual_async_fn`] no longer emits suggestions inside macros [#14142](https://github.com/rust-lang/rust-clippy/pull/14142)
* [`use_self`] skips analysis inside macro expansions of a `impl Self` block [#13128](https://github.com/rust-lang/rust-clippy/pull/13128)
* [`double_ended_iterator_last`] no longer triggers on non-reference immutable receiver [#14140](https://github.com/rust-lang/rust-clippy/pull/14140)

### ICE Fixes

* [`macro_use_imports`] Fix ICE when checking attributes
  [#14317](https://github.com/rust-lang/rust-clippy/pull/14317)
* [`doc_nested_refdefs`] Fix ICE by avoiding invalid ranges
  [#14308](https://github.com/rust-lang/rust-clippy/pull/14308)
* [`just_underscores_and_digits`] Fix ICE in error recovery scenario
  [#14168](https://github.com/rust-lang/rust-clippy/pull/14168)
* [`declare_interior_mutable_const`], [`borrow_interior_mutable_const`] Fix ICE by properly resolving `<T as Trait>::AssocT` projections
  [#14125](https://github.com/rust-lang/rust-clippy/pull/14125)

### Documentation Improvements

* [`struct_excessive_bools`] Documentation improved with rationale
  [#14351](https://github.com/rust-lang/rust-clippy/pull/14351)

### Others

* Use edition=2021 in `rustc_tools_util`
  [#14211](https://github.com/rust-lang/rust-clippy/pull/14211)
* Fix rustc_tools_util's `version.host_compiler` release channel, expose the rustc version, and add tests
  [#14123](https://github.com/rust-lang/rust-clippy/pull/14123)
* Make UI test annotations mandatory
  [#11421](https://github.com/rust-lang/rust-clippy/pull/11421)
  [#14388](https://github.com/rust-lang/rust-clippy/pull/14388)
  [#14393](https://github.com/rust-lang/rust-clippy/pull/14393)

## Rust 1.86

Current stable, released 2025-04-03

[View all 108 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2024-12-27T15%3A11%3A38Z..2025-02-06T13%3A57%3A58Z+base%3Amaster)

### New Lints

* Added [`unneeded_struct_pattern`] to `style` [#13465](https://github.com/rust-lang/rust-clippy/pull/13465)
* Added [`doc_overindented_list_items`] to `style` [#13711](https://github.com/rust-lang/rust-clippy/pull/13711)
* Added [`manual_ok_err`] to `complexity` [#13740](https://github.com/rust-lang/rust-clippy/pull/13740)
* Added [`non_std_lazy_statics`] to `pedantic` [#13770](https://github.com/rust-lang/rust-clippy/pull/13770)
* Added [`manual_repeat_n`] to `style` [#13858](https://github.com/rust-lang/rust-clippy/pull/13858)
* Added [`manual_option_as_slice`] to `complexity` [#13901](https://github.com/rust-lang/rust-clippy/pull/13901)
* Added [`double_ended_iterator_last`] to `perf` [#13922](https://github.com/rust-lang/rust-clippy/pull/13922)
* Added [`useless_nonzero_new_unchecked`] to `complexity` [#13993](https://github.com/rust-lang/rust-clippy/pull/13993)
* Added [`sliced_string_as_bytes`] to `perf` [#14002](https://github.com/rust-lang/rust-clippy/pull/14002)
* Added [`unnecessary_semicolon`] to `pedantic` [#14032](https://github.com/rust-lang/rust-clippy/pull/14032)
* Added [`return_and_then`] to `restriction` [#14051](https://github.com/rust-lang/rust-clippy/pull/14051)
* Added [`manual_slice_fill`] to `style` [#14082](https://github.com/rust-lang/rust-clippy/pull/14082)
* Added [`precedence_bits`] to `restriction` [#14115](https://github.com/rust-lang/rust-clippy/pull/14115)

### Moves and Deprecations

* Moved [`redundant_locals`] to `suspicious` (from `correctness`, now warn-by-default)
  [#13747](https://github.com/rust-lang/rust-clippy/pull/13747)
* Moved [`format_push_string`] to `pedantic` (from `restriction`)
  [#13894](https://github.com/rust-lang/rust-clippy/pull/13894)
* Moved [`format_collect`] to `pedantic` (from `perf`, now allow-by-default)
  [#13894](https://github.com/rust-lang/rust-clippy/pull/13894)
* Moved [`mutex_integer`] to `restriction` (from `nursery`) [#14110](https://github.com/rust-lang/rust-clippy/pull/14110)

### Enhancements

* Add `lint-inconsistent-struct-field-initializers` configuration option to [`inconsistent_struct_constructor`]
  [#13737](https://github.com/rust-lang/rust-clippy/pull/13737)
* [`len_zero`] now also triggers if deref target implements `is_empty()`
  [#13871](https://github.com/rust-lang/rust-clippy/pull/13871)
* [`obfuscated_if_else`] now also triggers for the `.then(..).unwrap_or(..)` pattern
  [#14021](https://github.com/rust-lang/rust-clippy/pull/14021)

### False Positive Fixes

* [`trailing_empty_array`] no longer triggers in tests [#13844](https://github.com/rust-lang/rust-clippy/pull/13844)
* [`missing_const_for_fn`] no longer triggers in tests [#13945](https://github.com/rust-lang/rust-clippy/pull/13945)
* [`significant_drop_in_scrutinee`]: do not falsely warn for temporaries created by `.await` expansion
  [#13985](https://github.com/rust-lang/rust-clippy/pull/13985)

### ICE Fixes

* [`borrow_interior_mutable_const`] Fix an ICE that can occur when taking a reference to a tuple/`struct` field of an
  interior mutable `const` [#13877](https://github.com/rust-lang/rust-clippy/pull/13877)

### Others

* Clippy now uses Rust edition 2024 [#13751](https://github.com/rust-lang/rust-clippy/pull/13751)

## Rust 1.85

Released 2025-02-20

[View all 72 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2024-11-15T19%3A31%3A08Z..2024-12-26T13%3A59%3A48Z+base%3Amaster)

### New Lints

* Added [`repr_packed_without_abi`] to `suspicious`
  [#13398](https://github.com/rust-lang/rust-clippy/pull/13398)
* Added [`as_pointer_underscore`] to `restriction`
  [#13251](https://github.com/rust-lang/rust-clippy/pull/13251)
* Added [`doc_nested_refdefs`] to `suspicious`
  [#13707](https://github.com/rust-lang/rust-clippy/pull/13707)
* Added [`literal_string_with_formatting_args`] to `nursery`
  [#13410](https://github.com/rust-lang/rust-clippy/pull/13410)
* Added [`doc_include_without_cfg`] to `restriction`
  [#13625](https://github.com/rust-lang/rust-clippy/pull/13625)

### Enhancements

* [`indexing_slicing`]: Can now be allowed in tests using the [`allow-indexing-slicing-in-tests`]
  configuration
  [#13854](https://github.com/rust-lang/rust-clippy/pull/13854)
* [`if_let_mutex`]: disable lint from Edition 2024 since
  [if_let_rescope](https://github.com/rust-lang/rust/issues/131154) was stabilized
  [#13695](https://github.com/rust-lang/rust-clippy/pull/13695)
* [`format_in_format_args`], [`recursive_format_impl`], [`to_string_in_format_args`],
  [`uninlined_format_args`], [`unused_format_specs`]: Can now support 3rd party format macros
  if they're marked with the `#[clippy::format_args]` attribute
  [#9948](https://github.com/rust-lang/rust-clippy/pull/9948)

### ICE Fixes

* [`trait_duplication_in_bounds`]: fix ICE on duplicate type or constant bound
  [#13722](https://github.com/rust-lang/rust-clippy/pull/13722)

### Others

* `clippy_utils` is now published to crates.io. Note that this crate is and will remain unstable.
  [#13700](https://github.com/rust-lang/rust-clippy/pull/13700)

## Rust 1.84

Released 2025-01-09

[View all 84 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2024-10-03T21%3A23%3A58Z..2024-11-14T17%3A41%3A37Z+base%3Amaster)

### New Lints

* Added [`unnecessary_map_or`] to `style`
  [#11796](https://github.com/rust-lang/rust-clippy/pull/11796)
* Added [`arbitrary_source_item_ordering`] to `restriction`
  [#13376](https://github.com/rust-lang/rust-clippy/pull/13376)
* Added [`map_with_unused_argument_over_ranges`] to `restriction`
  [#13034](https://github.com/rust-lang/rust-clippy/pull/13034)
* Added [`map_all_any_identity`] to `complexity`
  [#13499](https://github.com/rust-lang/rust-clippy/pull/13499)
* Added [`needless_as_bytes`] to `complexity`
  [#13437](https://github.com/rust-lang/rust-clippy/pull/13437)
* Added [`unnecessary_literal_bound`] to `pedantic`
  [#13395](https://github.com/rust-lang/rust-clippy/pull/13395)
* Added [`manual_ignore_case_cmp`] to `perf`
  [#13334](https://github.com/rust-lang/rust-clippy/pull/13334)
* Added [`regex_creation_in_loops`] to `perf`
  [#13412](https://github.com/rust-lang/rust-clippy/pull/13412)

### Moves and Deprecations

* Moved [`manual_is_power_of_two`] to `pedantic` (From `complexity`, now allow-by-default)
  [#13553](https://github.com/rust-lang/rust-clippy/pull/13553)
* Move [`module_name_repetitions`] to `restriction` (from `pedantic`)
  [#13541](https://github.com/rust-lang/rust-clippy/pull/13541)

### Enhancements

* [`doc_markdown`]: Added the following identifiers to [`doc-valid-idents`]:
  CoAP, MHz, GHz, and THz
  [#13633](https://github.com/rust-lang/rust-clippy/pull/13633)
  [#13460](https://github.com/rust-lang/rust-clippy/pull/13460)
* [`large_const_arrays`]: Changed the default of [`array-size-threshold`] to `16kb` (from `512kb`)
  [#13485](https://github.com/rust-lang/rust-clippy/pull/13485)

## Rust 1.83

Released 2024-11-28

[View all 64 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2024-08-25T09%3A59%3A01Z..2024-10-03T13%3A42%3A56Z+base%3Amaster)

### Important Change

* Removed the implicit `cargo-clippy` feature set by Clippy as announced here:
  <https://blog.rust-lang.org/2024/02/28/Clippy-deprecating-feature-cargo-clippy.html>
  [#13246](https://github.com/rust-lang/rust-clippy/pull/13246)

### New Lints

* Added [`unused_trait_names`] to `restriction`
  [#13322](https://github.com/rust-lang/rust-clippy/pull/13322)
* Added [`unnecessary_first_then_check`] to `complexity`
  [#13421](https://github.com/rust-lang/rust-clippy/pull/13421)
* Added [`non_zero_suggestions`] to `restriction`
  [#13167](https://github.com/rust-lang/rust-clippy/pull/13167)
* Added [`manual_is_power_of_two`] to `pedantic`
  [#13327](https://github.com/rust-lang/rust-clippy/pull/13327)
* Added [`manual_div_ceil`] to `complexity`
  [#12987](https://github.com/rust-lang/rust-clippy/pull/12987)
* Added [`zombie_processes`] to `suspicious`
  [#11476](https://github.com/rust-lang/rust-clippy/pull/11476)
* Added [`used_underscore_items`] to `pedantic`
  [#13294](https://github.com/rust-lang/rust-clippy/pull/13294)

### Moves and Deprecations

* Moved [`ref_option`] to `pedantic` (From `nursery`)
  [#13469](https://github.com/rust-lang/rust-clippy/pull/13469)
* Moved [`manual_c_str_literals`] to `complexity` (From `pedantic`, now warn-by-default)
  [#13263](https://github.com/rust-lang/rust-clippy/pull/13263)
* Moved [`empty_line_after_doc_comments`] to `suspicious` (From `nursery`, now warn-by-default)
  [#13091](https://github.com/rust-lang/rust-clippy/pull/13091)
* Moved [`empty_line_after_outer_attr`] to `suspicious` (From `nursery`, now warn-by-default)
  [#13091](https://github.com/rust-lang/rust-clippy/pull/13091)

### Enhancements

* [`missing_panics_doc`]: No longer lints in const environments
  [#13382](https://github.com/rust-lang/rust-clippy/pull/13382)

## Rust 1.82

Released 2024-10-17

[View all 108 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2024-07-11T20%3A12%3A07Z..2024-08-24T20%3A55%3A35Z+base%3Amaster)

### New Lints

* Added [`too_long_first_doc_paragraph`] to `nursery`
  [#12993](https://github.com/rust-lang/rust-clippy/pull/12993)
* Added [`unused_result_ok`] to `restriction`
  [#12150](https://github.com/rust-lang/rust-clippy/pull/12150)
* Added [`pathbuf_init_then_push`] to `restriction`
  [#11700](https://github.com/rust-lang/rust-clippy/pull/11700)

### Enhancements

* [`explicit_iter_loop`]: Now respects the `msrv` configuration
  [#13288](https://github.com/rust-lang/rust-clippy/pull/13288)
* [`assigning_clones`]: No longer lints in test code
  [#13273](https://github.com/rust-lang/rust-clippy/pull/13273)
* [`inconsistent_struct_constructor`]: Lint attributes now work on the struct definition
  [#13211](https://github.com/rust-lang/rust-clippy/pull/13211)
* [`set_contains_or_insert`]: Now also checks for `BTreeSet`
  [#13053](https://github.com/rust-lang/rust-clippy/pull/13053)
* [`doc_markdown`]: Added the following identifiers to [`doc-valid-idents`]: AccessKit,
  CoreFoundation, CoreGraphics, CoreText, Direct2D, Direct3D, DirectWrite, PostScript,
  OpenAL, OpenType, WebRTC, WebSocket, WebTransport, NetBSD, and OpenBSD
  [#13093](https://github.com/rust-lang/rust-clippy/pull/13093)

### ICE Fixes

* [`uninit_vec`]
  [rust#128720](https://github.com/rust-lang/rust/pull/128720)

## Rust 1.81

Released 2024-09-05

### New Lints

* Added [`cfg_not_test`] to `restriction`
  [#11293](https://github.com/rust-lang/rust-clippy/pull/11293)
* Added [`byte_char_slices`] to `style`
  [#10155](https://github.com/rust-lang/rust-clippy/pull/10155)
* Added [`set_contains_or_insert`] to `nursery`
  [#12873](https://github.com/rust-lang/rust-clippy/pull/12873)
* Added [`manual_rotate`] to `style`
  [#12983](https://github.com/rust-lang/rust-clippy/pull/12983)
* Added [`unnecessary_min_or_max`] to `complexity`
  [#12368](https://github.com/rust-lang/rust-clippy/pull/12368)
* Added [`manual_inspect`] to `complexity`
  [#12287](https://github.com/rust-lang/rust-clippy/pull/12287)
* Added [`field_scoped_visibility_modifiers`] to `restriction`
  [#12893](https://github.com/rust-lang/rust-clippy/pull/12893)
* Added [`manual_pattern_char_comparison`] to `style`
  [#12849](https://github.com/rust-lang/rust-clippy/pull/12849)
* Added [`needless_maybe_sized`] to `suspicious`
  [#10632](https://github.com/rust-lang/rust-clippy/pull/10632)
* Added [`needless_character_iteration`] to `suspicious`
  [#12815](https://github.com/rust-lang/rust-clippy/pull/12815)

### Moves and Deprecations

* [`allow_attributes`], [`allow_attributes_without_reason`]: Now work on stable
  [rust#120924](https://github.com/rust-lang/rust/pull/120924)
* Renamed `overflow_check_conditional` to [`panicking_overflow_checks`]
  [#12944](https://github.com/rust-lang/rust-clippy/pull/12944)
* Moved [`panicking_overflow_checks`] to `correctness` (From `complexity` now deny-by-default)
  [#12944](https://github.com/rust-lang/rust-clippy/pull/12944)
* Renamed `thread_local_initializer_can_be_made_const` to [`missing_const_for_thread_local`]
  [#12974](https://github.com/rust-lang/rust-clippy/pull/12974)
* Deprecated [`maybe_misused_cfg`] and [`mismatched_target_os`] as they are now caught by cargo
  and rustc
  [#12875](https://github.com/rust-lang/rust-clippy/pull/12875)

### Enhancements

* [`significant_drop_in_scrutinee`]: Now also checks scrutinies of `while let` and `for let`
  expressions
  [#12870](https://github.com/rust-lang/rust-clippy/pull/12870)
* [`std_instead_of_core`]: Now respects the `msrv` configuration
  [#13168](https://github.com/rust-lang/rust-clippy/pull/13168)

### ICE Fixes

* [`suboptimal_flops`]: No longer crashes on custom `.log()` functions
  [#12884](https://github.com/rust-lang/rust-clippy/pull/12884)

## Rust 1.80

Current stable, released 2024-07-25

[View all 68 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2024-04-18T22%3A50%3A22Z..2024-05-30T08%3A26%3A18Z+base%3Amaster)

### New Lints

* Added [`while_float`] to `nursery`
  [#12765](https://github.com/rust-lang/rust-clippy/pull/12765)
* Added [`macro_metavars_in_unsafe`] to `suspicious`
  [#12107](https://github.com/rust-lang/rust-clippy/pull/12107)
* Added [`renamed_function_params`] to `restriction`
  [#11540](https://github.com/rust-lang/rust-clippy/pull/11540)
* Added [`doc_lazy_continuation`] to `style`
  [#12770](https://github.com/rust-lang/rust-clippy/pull/12770)

### Moves and Deprecations

* Moved [`assigning_clones`] to `pedantic` (From `perf` now allow-by-default)
  [#12779](https://github.com/rust-lang/rust-clippy/pull/12779)
* Moved [`single_char_pattern`] to `pedantic` (From `perf` now allow-by-default)
  [#11852](https://github.com/rust-lang/rust-clippy/pull/11852)

### Enhancements

* [`panic`]: Added [`allow-panic-in-tests`] configuration to allow the lint in tests
  [#12803](https://github.com/rust-lang/rust-clippy/pull/12803)
* [`missing_const_for_fn`]: Now respects the [`msrv`] configuration
  [#12713](https://github.com/rust-lang/rust-clippy/pull/12713)
* [`missing_panics_doc`]: No longer lints on compile-time panics
  [#12790](https://github.com/rust-lang/rust-clippy/pull/12790)
* [`collapsible_match`]: Now considers the [`msrv`] configuration for the suggestion
  [#12745](https://github.com/rust-lang/rust-clippy/pull/12745)
* [`useless_vec`]: Added [`allow-useless-vec-in-tests`] configuration to allow the lint in tests
  [#12725](https://github.com/rust-lang/rust-clippy/pull/12725)

### Suggestion Fixes/Improvements

* [`single_match`], [`single_match_else`]: Suggestions are now machine-applicable
  [#12726](https://github.com/rust-lang/rust-clippy/pull/12726)

## Rust 1.79

Released 2024-06-13

[View all 102 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2024-03-08T11%3A13%3A58Z..2024-04-18T15%3A50%3A50Z+base%3Amaster)

### New Lints

* Added [`legacy_numeric_constants`] to `style`
  [#12312](https://github.com/rust-lang/rust-clippy/pull/12312)
* Added [`missing_transmute_annotations`] to `suspicious`
  [#12239](https://github.com/rust-lang/rust-clippy/pull/12239)
* Added [`integer_division_remainder_used`] to `restriction`
  [#12451](https://github.com/rust-lang/rust-clippy/pull/12451)
* Added [`duplicated_attributes`] to `suspicious`
  [#12378](https://github.com/rust-lang/rust-clippy/pull/12378)
* Added [`manual_unwrap_or_default`] to `suspicious`
  [#12440](https://github.com/rust-lang/rust-clippy/pull/12440)
* Added [`zero_repeat_side_effects`] to `suspicious`
  [#12449](https://github.com/rust-lang/rust-clippy/pull/12449)
* Added [`const_is_empty`] to `suspicious`
  [#12310](https://github.com/rust-lang/rust-clippy/pull/12310)

### Moves and Deprecations

* Moved [`box_default`] to `style` (From `perf`)
  [#12601](https://github.com/rust-lang/rust-clippy/pull/12601)
* Moved [`manual_clamp`] to `complexity` (From `nursery` now warn-by-default)
  [#12543](https://github.com/rust-lang/rust-clippy/pull/12543)
* Moved [`readonly_write_lock`] to `perf` (From `nursery` now warn-by-default)
  [#12479](https://github.com/rust-lang/rust-clippy/pull/12479)

### Enhancements

* [`module_name_repetitions`]: Added the [`allowed-prefixes`] configuration to allow common prefixes.
  [#12573](https://github.com/rust-lang/rust-clippy/pull/12573)
* [`cast_sign_loss`], [`cast_possible_truncation`], [`cast_lossless`]: Are now allowed in macros
  [#12631](https://github.com/rust-lang/rust-clippy/pull/12631)
* [`manual_clamp`]: Now only lints on constant min and max values
  [#12543](https://github.com/rust-lang/rust-clippy/pull/12543)
* [`assigning_clones`]: Now considers the [`msrv`] configuration
  [#12511](https://github.com/rust-lang/rust-clippy/pull/12511)
* [`needless_return`], [`useless_let_if_seq`], [`mut_mut`], [`read_zero_byte_vec`], [`unused_io_amount`],
  [`unused_peekable`]: Now respects `#[allow]` attributes on the affected statement instead
  [#12446](https://github.com/rust-lang/rust-clippy/pull/12446)

### False Positive Fixes

* [`cast_lossless`]: No longer lints when casting to `u128`
  [#12496](https://github.com/rust-lang/rust-clippy/pull/12496)
* [`std_instead_of_core`] No longer lints on modules that are only in `std`
  [#12447](https://github.com/rust-lang/rust-clippy/pull/12447)

### ICE Fixes

* [`needless_return`]: No longer crashes on non-ascii characters
  [#12493](https://github.com/rust-lang/rust-clippy/pull/12493)

## Rust 1.78

Released 2024-05-02

[View all 112 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2024-01-26T05%3A46%3A23Z..2024-03-07T16%3A25%3A52Z+base%3Amaster)

### New Lints

* [`assigning_clones`]
  [#12077](https://github.com/rust-lang/rust-clippy/pull/12077)
* [`mixed_attributes_style`]
  [#12354](https://github.com/rust-lang/rust-clippy/pull/12354)
* [`empty_docs`]
  [#12342](https://github.com/rust-lang/rust-clippy/pull/12342)
* [`unnecessary_get_then_check`]
  [#12339](https://github.com/rust-lang/rust-clippy/pull/12339)
* [`multiple_bound_locations`]
  [#12259](https://github.com/rust-lang/rust-clippy/pull/12259)
* [`unnecessary_clippy_cfg`]
  [#12303](https://github.com/rust-lang/rust-clippy/pull/12303)
* [`deprecated_clippy_cfg_attr`]
  [#12292](https://github.com/rust-lang/rust-clippy/pull/12292)
* [`manual_c_str_literals`]
  [#11919](https://github.com/rust-lang/rust-clippy/pull/11919)
* [`ref_as_ptr`]
  [#12087](https://github.com/rust-lang/rust-clippy/pull/12087)
* [`lint_groups_priority`]
  [#11832](https://github.com/rust-lang/rust-clippy/pull/11832)
* [`unnecessary_result_map_or_else`]
  [#12169](https://github.com/rust-lang/rust-clippy/pull/12169)
* [`to_string_trait_impl`]
  [#12122](https://github.com/rust-lang/rust-clippy/pull/12122)
* [`incompatible_msrv`]
  [#12160](https://github.com/rust-lang/rust-clippy/pull/12160)

### Enhancements

* [`thread_local_initializer_can_be_made_const`]: Now checks the [`msrv`] configuration
  [#12405](https://github.com/rust-lang/rust-clippy/pull/12405)
* [`disallowed_macros`]: Code generated by derive macros can no longer allow this lint
  [#12267](https://github.com/rust-lang/rust-clippy/pull/12267)
* [`wildcard_imports`]: Add configuration [`allowed-wildcard-imports`] to allow preconfigured wildcards
  [#11979](https://github.com/rust-lang/rust-clippy/pull/11979)

### ICE Fixes

* [`ptr_as_ptr`]: No longer ICEs when the cast source is a function call to a local variable
  [#12617](https://github.com/rust-lang/rust-clippy/pull/12617)
* [`cast_sign_loss`]: Avoids an infinite loop when casting two chained `.unwrap()` calls
  [#12508](https://github.com/rust-lang/rust-clippy/pull/12508)

## Rust 1.77

Released 2024-03-18

[View all 93 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2023-12-16T18%3A20%3A00Z..2024-01-25T18%3A15%3A56Z+base%3Amaster)

### New Lints

* [`suspicious_open_options`]
  [#11608](https://github.com/rust-lang/rust-clippy/pull/11608)
* [`option_as_ref_cloned`]
  [#12051](https://github.com/rust-lang/rust-clippy/pull/12051)
* [`thread_local_initializer_can_be_made_const`]
  [#12026](https://github.com/rust-lang/rust-clippy/pull/12026)
* [`str_split_at_newline`]
  [#11987](https://github.com/rust-lang/rust-clippy/pull/11987)
* [`empty_enum_variants_with_brackets`]
  [#12047](https://github.com/rust-lang/rust-clippy/pull/12047)
* [`manual_is_variant_and`]
  [#11865](https://github.com/rust-lang/rust-clippy/pull/11865)
* [`pub_underscore_fields`]
  [#10283](https://github.com/rust-lang/rust-clippy/pull/10283)
* [`eager_transmute`]
  [#11981](https://github.com/rust-lang/rust-clippy/pull/11981)
* [`iter_filter_is_some`]
  [#12004](https://github.com/rust-lang/rust/pull/12004)
* [`iter_filter_is_ok`]
  [#12004](https://github.com/rust-lang/rust/pull/12004)
* [`result_filter_map`]
  [#11869](https://github.com/rust-lang/rust-clippy/pull/11869)
* [`unconditional_recursion`]
  [#11938](https://github.com/rust-lang/rust-clippy/pull/11938)

### Enhancements

* [`multiple_crate_versions`]: Added the [`allowed-duplicate-crates`] configuration to allow specific crates
  [#12179](https://github.com/rust-lang/rust-clippy/pull/12179)
* [`single_call_fn`]: No longer ignores `#[allow]` attributes
  [#12183](https://github.com/rust-lang/rust-clippy/pull/12183)
* [`read_zero_byte_vec`]: Updated the heuristics used for linting
  [#11766](https://github.com/rust-lang/rust-clippy/pull/11766)

### ICE Fixes

* [`unit_arg`]: No longer crashes when checking for const in nested bodies
  [#11977](https://github.com/rust-lang/rust-clippy/pull/11977)
* [`indexing_slicing`]: No longer crashes when the array index exceeds `usize`
  [#12266](https://github.com/rust-lang/rust-clippy/pull/12266)

### Others

* Warnings about invalid fields inside `clippy.toml` files now include suggestions for existing fields
  [#12180](https://github.com/rust-lang/rust-clippy/pull/12180)

## Rust 1.76

Released 2024-02-08

[View all 85 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2023-11-02T20%3A23%3A40Z..2023-12-16T13%3A11%3A08Z+base%3Amaster)

### New Lints

- [`infinite_loop`]
  [#11829](https://github.com/rust-lang/rust-clippy/pull/11829)
- [`ineffective_open_options`]
  [#11902](https://github.com/rust-lang/rust-clippy/pull/11902)
- [`uninhabited_references`]
  [#11878](https://github.com/rust-lang/rust-clippy/pull/11878)
- [`repeat_vec_with_capacity`]
  [#11597](https://github.com/rust-lang/rust-clippy/pull/11597)
- [`test_attr_in_doctest`]
  [#11872](https://github.com/rust-lang/rust-clippy/pull/11872)
- [`option_map_or_err_ok`]
  [#11864](https://github.com/rust-lang/rust-clippy/pull/11864)
- [`join_absolute_paths`]
  [#11453](https://github.com/rust-lang/rust-clippy/pull/11453)
- [`impl_hash_borrow_with_str_and_bytes`]
  [#11781](https://github.com/rust-lang/rust-clippy/pull/11781)
- [`iter_over_hash_type`]
  [#11791](https://github.com/rust-lang/rust-clippy/pull/11791)

### Moves and Deprecations

- Renamed `blocks_in_if_conditions` to [`blocks_in_conditions`]
  [#11853](https://github.com/rust-lang/rust-clippy/pull/11853)
- Moved [`implied_bounds_in_impls`] to `complexity` (Now warn-by-default)
  [#11867](https://github.com/rust-lang/rust-clippy/pull/11867)
- Moved [`if_same_then_else`] to `style` (Now warn-by-default)
  [#11809](https://github.com/rust-lang/rust-clippy/pull/11809)

### Enhancements

- [`missing_safety_doc`], [`unnecessary_safety_doc`], [`missing_panics_doc`], [`missing_errors_doc`]:
  Added the [`check-private-items`] configuration to enable lints on private items
  [#11842](https://github.com/rust-lang/rust-clippy/pull/11842)

### ICE Fixes

- [`impl_trait_in_params`]: No longer crashes when a function has generics but no function parameters
  [#11804](https://github.com/rust-lang/rust-clippy/pull/11804)
- [`unused_enumerate_index`]: No longer crashes on empty tuples
  [#11756](https://github.com/rust-lang/rust-clippy/pull/11756)

### Others

- Clippy now respects the `CARGO` environment value
  [#11944](https://github.com/rust-lang/rust-clippy/pull/11944)

## Rust 1.75

Released 2023-12-28

[View all 69 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2023-09-25T11%3A47%3A47Z..2023-11-02T16%3A41%3A59Z+base%3Amaster)

### New Lints

* [`unused_enumerate_index`]
  [#10404](https://github.com/rust-lang/rust-clippy/pull/10404)
* [`unnecessary_fallible_conversions`]
  [#11669](https://github.com/rust-lang/rust-clippy/pull/11669)
* [`waker_clone_wake`]
  [#11698](https://github.com/rust-lang/rust-clippy/pull/11698)
* [`struct_field_names`]
  [#11496](https://github.com/rust-lang/rust-clippy/pull/11496)
* [`into_iter_without_iter`]
  [#11587](https://github.com/rust-lang/rust-clippy/pull/11587)
* [`iter_without_into_iter`]
  [#11527](https://github.com/rust-lang/rust-clippy/pull/11527)
* [`manual_hash_one`]
  [#11556](https://github.com/rust-lang/rust-clippy/pull/11556)


### Moves and Deprecations

* Moved [`read_zero_byte_vec`] to `nursery` (Now allow-by-default)
  [#11727](https://github.com/rust-lang/rust-clippy/pull/11727)
* Moved [`missing_enforced_import_renames`] to `style` (Now warn-by-default)
  [#11539](https://github.com/rust-lang/rust-clippy/pull/11539)
* Moved [`needless_raw_string_hashes`] to `pedantic` (Now allow-by-default)
  [#11415](https://github.com/rust-lang/rust-clippy/pull/11415)
* Moved [`needless_pass_by_ref_mut`] to `nursery` (Now allow-by-default)
  [#11596](https://github.com/rust-lang/rust-clippy/pull/11596)

### Enhancements

* [`declare_interior_mutable_const`] and [`borrow_interior_mutable_const`]: Now check the
  [`ignore-interior-mutability`] config value
  [#11678](https://github.com/rust-lang/rust-clippy/pull/11678)

### Suggestion Fixes/Improvements

* [`items_after_test_module`]: The suggestion is now machine-applicable
  [#11611](https://github.com/rust-lang/rust-clippy/pull/11611)

### ICE Fixes

* [`redundant_locals`]: No longer crashes if variables are rebound above macros
  [#11623](https://github.com/rust-lang/rust-clippy/pull/11623)
* [`implicit_hasher`]: No longer lints inside macros, which could cause ICEs
  [#11593](https://github.com/rust-lang/rust-clippy/pull/11593)

### Documentation Improvements

* `cargo clippy --help` now uses colors for readability :tada:

## Rust 1.74

Released 2023-11-16

[View all 94 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2023-08-11T15%3A29%3A18Z..2023-09-25T08%3A48%3A22Z+base%3Amaster)

### New Lints

* [`redundant_as_str`]
  [#11526](https://github.com/rust-lang/rust-clippy/pull/11526)
* [`needless_borrows_for_generic_args`]
  [#11511](https://github.com/rust-lang/rust-clippy/pull/11511)
* [`path_ends_with_ext`]
  [#11483](https://github.com/rust-lang/rust-clippy/pull/11483)
* [`unnecessary_map_on_constructor`]
  [#11413](https://github.com/rust-lang/rust-clippy/pull/11413)
* [`missing_asserts_for_indexing`]
  [#10692](https://github.com/rust-lang/rust-clippy/pull/10692)
* [`iter_out_of_bounds`]
  [#11396](https://github.com/rust-lang/rust-clippy/pull/11396)
* [`implied_bounds_in_impls`]
  [#11362](https://github.com/rust-lang/rust-clippy/pull/11362)
* [`reserve_after_initialization`]
  [#11373](https://github.com/rust-lang/rust-clippy/pull/11373)
* [`should_panic_without_expect`]
  [#11204](https://github.com/rust-lang/rust-clippy/pull/11204)

### Moves and Deprecations

* Renamed `incorrect_clone_impl_on_copy_type` to [`non_canonical_clone_impl`]
  [#11358](https://github.com/rust-lang/rust-clippy/pull/11358)
* Renamed `incorrect_partial_ord_impl_on_ord_type` to [`non_canonical_partial_ord_impl`]
  [#11358](https://github.com/rust-lang/rust-clippy/pull/11358)
* Moved [`non_canonical_clone_impl`] to `suspicious` (Now warn-by-default)
  [#11358](https://github.com/rust-lang/rust-clippy/pull/11358)
* Moved [`non_canonical_partial_ord_impl`] to `suspicious` (Now warn-by-default)
  [#11358](https://github.com/rust-lang/rust-clippy/pull/11358)
* Moved [`needless_pass_by_ref_mut`] to `nursery` (Now allow-by-default)
  [#11596](https://github.com/rust-lang/rust-clippy/pull/11596)

### Enhancements

* [`undocumented_unsafe_blocks`]: The config values [`accept-comment-above-statement`] and 
  [`accept-comment-above-attributes`] are now `true` by default
  [#11170](https://github.com/rust-lang/rust-clippy/pull/11170)
* [`explicit_iter_loop`]: Added [`enforce-iter-loop-reborrow`] to disable reborrow linting by default
  [#11418](https://github.com/rust-lang/rust-clippy/pull/11418)

### ICE Fixes

* [`enum_variant_names`]: No longer crashes if the threshold is 0 and the enum has no variants
  [#11552](https://github.com/rust-lang/rust-clippy/pull/11552)
* [`cast_possible_truncation`]: No longer crashes on values larger than `u64::MAX`
  [#11517](https://github.com/rust-lang/rust-clippy/pull/11517)
* [`tuple_array_conversions`]: No longer crashes if the array length is not usize
  [#11379](https://github.com/rust-lang/rust-clippy/pull/11379)
* [`useless_conversion`]: No longer crashes, when the receiver is a non-fn item
  [#11070](https://github.com/rust-lang/rust-clippy/pull/11070)

## Rust 1.73

Released 2023-10-05

[View all 103 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2023-07-02T12%3A24%3A40Z..2023-08-11T11%3A09%3A56Z+base%3Amaster)

### New Lints

* [`impossible_comparisons`]
  [#10843](https://github.com/rust-lang/rust-clippy/pull/10843)
* [`redundant_comparisons`]
  [#10843](https://github.com/rust-lang/rust-clippy/pull/10843)
* [`ignored_unit_patterns`]
  [#11242](https://github.com/rust-lang/rust-clippy/pull/11242)
* [`readonly_write_lock`]
  [#11210](https://github.com/rust-lang/rust-clippy/pull/11210)
* [`filter_map_bool_then`]
  [#11115](https://github.com/rust-lang/rust-clippy/pull/11115)
* [`needless_return_with_question_mark`]
  [#11031](https://github.com/rust-lang/rust-clippy/pull/11031)
* [`redundant_guards`]
  [#10955](https://github.com/rust-lang/rust-clippy/pull/10955)
* [`redundant_locals`]
  [#10885](https://github.com/rust-lang/rust-clippy/pull/10885)
* [`absolute_paths`]
  [#11003](https://github.com/rust-lang/rust-clippy/pull/11003)
* [`error_impl_error`]
  [#11107](https://github.com/rust-lang/rust-clippy/pull/11107)
* [`iter_skip_zero`]
  [#11046](https://github.com/rust-lang/rust-clippy/pull/11046)
* [`string_lit_chars_any`]
  [#11052](https://github.com/rust-lang/rust-clippy/pull/11052)
* [`four_forward_slashes`]
  [#11140](https://github.com/rust-lang/rust-clippy/pull/11140)
* [`format_collect`]
  [#11116](https://github.com/rust-lang/rust-clippy/pull/11116)
* [`needless_pass_by_ref_mut`]
  [#10900](https://github.com/rust-lang/rust-clippy/pull/10900)
* [`manual_is_infinite`]
  [#11049](https://github.com/rust-lang/rust-clippy/pull/11049)
* [`manual_is_finite`]
  [#11049](https://github.com/rust-lang/rust-clippy/pull/11049)
* [`incorrect_partial_ord_impl_on_ord_type`]
  [#10788](https://github.com/rust-lang/rust-clippy/pull/10788)
* [`read_line_without_trim`]
  [#10970](https://github.com/rust-lang/rust-clippy/pull/10970)
* [`type_id_on_box`]
  [#10987](https://github.com/rust-lang/rust-clippy/pull/10987)

### Moves and Deprecations

* Renamed `unwrap_or_else_default` to [`unwrap_or_default`]
  [#10120](https://github.com/rust-lang/rust-clippy/pull/10120)
* Moved [`tuple_array_conversions`] to `pedantic` (Now allow-by-default)
  [#11146](https://github.com/rust-lang/rust-clippy/pull/11146)
* Moved [`arc_with_non_send_sync`] to `suspicious` (Now warn-by-default)
  [#11104](https://github.com/rust-lang/rust-clippy/pull/11104)
* Moved [`needless_raw_string_hashes`] to `pedantic` (Now allow-by-default)
  [#11415](https://github.com/rust-lang/rust-clippy/pull/11415)

### Enhancements

* [`unwrap_used`]: No longer lints on the never-type or never-like enums
  [#11252](https://github.com/rust-lang/rust-clippy/pull/11252)
* [`expect_used`]: No longer lints on the never-type or never-like enums
  [#11252](https://github.com/rust-lang/rust-clippy/pull/11252)

### False Positive Fixes

* [`panic_in_result_fn`]: No longer triggers on `todo!`, `unimplemented!`, `unreachable!`
  [#11123](https://github.com/rust-lang/rust-clippy/pull/11123)

### Suggestion Fixes/Improvements

* [`semicolon_if_nothing_returned`]: The suggestion is now machine-applicable with rustfix
  [#11083](https://github.com/rust-lang/rust-clippy/pull/11083)

### ICE Fixes

* [`filter_map_bool_then`]: No longer crashes on late-bound regions
  [#11318](https://github.com/rust-lang/rust-clippy/pull/11318)
* [`unwrap_or_default`]: No longer crashes on alias types for local items
  [#11258](https://github.com/rust-lang/rust-clippy/pull/11258)
* [`unnecessary_literal_unwrap`]: No longer crashes on `None.unwrap_or_default()`
  [#11106](https://github.com/rust-lang/rust-clippy/pull/11106)
* Fixed MIR-related ICE
  [#11130](https://github.com/rust-lang/rust-clippy/pull/11130)
* [`missing_fields_in_debug`]: No longer crashes on non-ADT self types
  [#11069](https://github.com/rust-lang/rust-clippy/pull/11069)

## Rust 1.72

Released 2023-08-24

[View all 131 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2023-05-22T14%3A53%3A59Z..2023-07-01T22%3A57%3A20Z+base%3Amaster)

### New Lints

* [`manual_try_fold`]
  [#11012](https://github.com/rust-lang/rust-clippy/pull/11012)
* [`tuple_array_conversions`]
  [#11020](https://github.com/rust-lang/rust-clippy/pull/11020)
* [`redundant_at_rest_pattern`]
  [#11013](https://github.com/rust-lang/rust-clippy/pull/11013)
* [`needless_pub_self`]
  [#10967](https://github.com/rust-lang/rust-clippy/pull/10967)
* [`pub_with_shorthand`]
  [#10967](https://github.com/rust-lang/rust-clippy/pull/10967)
* [`pub_without_shorthand`]
  [#10967](https://github.com/rust-lang/rust-clippy/pull/10967)
* [`manual_range_patterns`]
  [#10968](https://github.com/rust-lang/rust-clippy/pull/10968)
* [`needless_raw_string_hashes`]
  [#10884](https://github.com/rust-lang/rust-clippy/pull/10884)
* [`needless_raw_strings`]
  [#10884](https://github.com/rust-lang/rust-clippy/pull/10884)
* [`incorrect_clone_impl_on_copy_type`]
  [#10925](https://github.com/rust-lang/rust-clippy/pull/10925)
* [`drain_collect`]
  [#10835](https://github.com/rust-lang/rust-clippy/pull/10835)
* [`single_range_in_vec_init`]
  [#10934](https://github.com/rust-lang/rust-clippy/pull/10934)
* [`unnecessary_literal_unwrap`]
  [#10358](https://github.com/rust-lang/rust-clippy/pull/10358)
* [`large_stack_frames`]
  [#10827](https://github.com/rust-lang/rust-clippy/pull/10827)
* [`min_ident_chars`]
  [#10916](https://github.com/rust-lang/rust-clippy/pull/10916)
* [`needless_if`]
  [#10921](https://github.com/rust-lang/rust-clippy/pull/10921)
* [`excessive_nesting`]
  [#10672](https://github.com/rust-lang/rust-clippy/pull/10672)
* [`arc_with_non_send_sync`]
  [#10898](https://github.com/rust-lang/rust-clippy/pull/10898)
* [`redundant_type_annotations`]
  [#10570](https://github.com/rust-lang/rust-clippy/pull/10570)
* [`host_endian_bytes`]
  [#10826](https://github.com/rust-lang/rust-clippy/pull/10826)
* [`little_endian_bytes`]
  [#10826](https://github.com/rust-lang/rust-clippy/pull/10826)
* [`big_endian_bytes`]
  [#10826](https://github.com/rust-lang/rust-clippy/pull/10826)
* [`ptr_cast_constness`]
  [#10779](https://github.com/rust-lang/rust-clippy/pull/10779)
* [`needless_else`]
  [#10810](https://github.com/rust-lang/rust-clippy/pull/10810)

### Moves and Deprecations

* Moved [`redundant_clone`] to `nursery` (Now allow-by-default)
  [#10873](https://github.com/rust-lang/rust-clippy/pull/10873)

### Enhancements

* [`undocumented_unsafe_blocks`]: Added [`accept-comment-above-attributes`] configuration
  [#10986](https://github.com/rust-lang/rust-clippy/pull/10986)
* [`undocumented_unsafe_blocks`]: Added [`accept-comment-above-statement`] configuration.
  [#10886](https://github.com/rust-lang/rust-clippy/pull/10886)
* [`missing_panics_doc`]: No longer lints on `todo!()`
  [#10976](https://github.com/rust-lang/rust-clippy/pull/10976)
* [`module_inception`]: Added `allow-private-module-inception` configuration.
  [#10917](https://github.com/rust-lang/rust-clippy/pull/10917)
* Errors and warnings generated while parsing `clippy.toml` now point to the location in the TOML
  file the error/warning occurred in.
  [#10607](https://github.com/rust-lang/rust-clippy/pull/10607)

### False Positive Fixes

* [`excessive_precision`]: No longer lints overflowing literals
  [#10952](https://github.com/rust-lang/rust-clippy/pull/10952)

### Suggestion Fixes/Improvements

* [`option_map_unwrap_or`]: The suggestion now considers the set [`msrv`] config value
  [#11030](https://github.com/rust-lang/rust-clippy/pull/11030)

### Documentation Improvements

* [Clippy's lint list] now stores filter parameters in the URL, to allow easy sharing
  [#10834](https://github.com/rust-lang/rust-clippy/pull/10834)

## Rust 1.71

Released 2023-07-13

Note: Clippy will use a shorter changelog format from now on, if you want a detailed list of
all changes, please check out the list of merged pull requests.

[View all 78 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2023-04-11T20%3A05%3A26Z..2023-05-20T13%3A48%3A17Z+base%3Amaster)

### New Lints

* [`non_minimal_cfg`]
  [#10763](https://github.com/rust-lang/rust-clippy/pull/10763)
* [`manual_next_back`]
  [#10769](https://github.com/rust-lang/rust-clippy/pull/10769)
* [`ref_patterns`]
  [#10736](https://github.com/rust-lang/rust-clippy/pull/10736)
* [`default_constructed_unit_structs`]
  [#10716](https://github.com/rust-lang/rust-clippy/pull/10716)
* [`manual_while_let_some`]
  [#10647](https://github.com/rust-lang/rust-clippy/pull/10647)
* [`needless_bool_assign`]
  [#10432](https://github.com/rust-lang/rust-clippy/pull/10432)
* [`items_after_test_module`]
  [#10578](https://github.com/rust-lang/rust-clippy/pull/10578)

### Moves and Deprecations

* Rename `integer_arithmetic` to `arithmetic_side_effects`
  [#10674](https://github.com/rust-lang/rust-clippy/pull/10674)
* Moved [`redundant_clone`] to `nursery` (Now allow-by-default)
  [#10873](https://github.com/rust-lang/rust-clippy/pull/10873)

### Enhancements

* [`invalid_regex`]: Now supports the new syntax introduced after regex v1.8.0
  [#10682](https://github.com/rust-lang/rust-clippy/pull/10682)
* [`semicolon_outside_block`]: Added [`semicolon-outside-block-ignore-multiline`] as a new config value.
  [#10656](https://github.com/rust-lang/rust-clippy/pull/10656)
* [`semicolon_inside_block`]: Added [`semicolon-inside-block-ignore-singleline`] as a new config value.
  [#10656](https://github.com/rust-lang/rust-clippy/pull/10656)
* [`unnecessary_box_returns`]: Added [`unnecessary-box-size`] as a new config value to set the maximum
  size of `T` in `Box<T>` to be linted.
  [#10651](https://github.com/rust-lang/rust-clippy/pull/10651)

### Documentation Improvements

* `cargo clippy --explain LINT` now shows possible configuration options for the explained lint
  [#10751](https://github.com/rust-lang/rust-clippy/pull/10751)
* New config values mentioned in this changelog will now be linked.
  [#10889](https://github.com/rust-lang/rust-clippy/pull/10889)
* Several sections of [Clippy's book] have been reworked
  [#10652](https://github.com/rust-lang/rust-clippy/pull/10652)
  [#10622](https://github.com/rust-lang/rust-clippy/pull/10622)

[Clippy's book]: https://doc.rust-lang.org/clippy/

## Rust 1.70

Released 2023-06-01

[View all 91 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2023-02-26T01%3A05%3A43Z..2023-04-11T13%3A27%3A30Z+base%3Amaster)

### New Lints

* [`large_futures`]
  [#10414](https://github.com/rust-lang/rust-clippy/pull/10414)
* [`missing_assert_message`]
  [#10362](https://github.com/rust-lang/rust-clippy/pull/10362)
* [`clear_with_drain`]
  [#10528](https://github.com/rust-lang/rust-clippy/pull/10528)
* [`redundant_async_block`]
  [#10448](https://github.com/rust-lang/rust-clippy/pull/10448)
* [`collection_is_never_read`]
  [#10415](https://github.com/rust-lang/rust-clippy/pull/10415)
* [`let_with_type_underscore`]
  [#10467](https://github.com/rust-lang/rust-clippy/pull/10467)
* [`tests_outside_test_module`]
  [#10543](https://github.com/rust-lang/rust-clippy/pull/10543)
* [`allow_attributes`]
  [#10481](https://github.com/rust-lang/rust-clippy/pull/10481)
* [`suspicious_doc_comments`]
  [#10497](https://github.com/rust-lang/rust-clippy/pull/10497)
* [`unnecessary_box_returns`]
  [#9102](https://github.com/rust-lang/rust-clippy/pull/9102)
* [`manual_main_separator_str`]
  [#10483](https://github.com/rust-lang/rust-clippy/pull/10483)
* [`unnecessary_struct_initialization`]
  [#10489](https://github.com/rust-lang/rust-clippy/pull/10489)
* [`manual_slice_size_calculation`]
  [#10601](https://github.com/rust-lang/rust-clippy/pull/10601)
* [`lines_filter_map_ok`]
  [#10534](https://github.com/rust-lang/rust-clippy/pull/10534)

### Moves and Deprecations

* Moved [`let_underscore_untyped`] to `restriction`
  [#10442](https://github.com/rust-lang/rust-clippy/pull/10442)

### Enhancements

* [`extra_unused_type_parameters`]: No longer lints on public items if `avoid-breaking-exported-api` is set
  [#10536](https://github.com/rust-lang/rust-clippy/pull/10536)
* [`len_without_is_empty`]: Now also detects `async` functions
  [#10359](https://github.com/rust-lang/rust-clippy/pull/10359)
* [`arithmetic_side_effects`]: Now correctly handles divisions and modulo expressions if the right-hand-side
  is unknown
  [#10585](https://github.com/rust-lang/rust-clippy/pull/10585)
* [`nonminimal_bool`]: No longer ignores `#[allow]` attributes
  [#10588](https://github.com/rust-lang/rust-clippy/pull/10588)
* [`uninit_vec`], [`uninit_assumed_init`]: Now uses a better heuristic
  [#10520](https://github.com/rust-lang/rust-clippy/pull/10520)
* [`ifs_same_cond`]: Now also detects immutable method calls.
  [#10350](https://github.com/rust-lang/rust-clippy/pull/10350)
* [`arithmetic_side_effects`]: No longer lints on right or left shifts with constant integers, as the
  compiler warns about them
  [#10309](https://github.com/rust-lang/rust-clippy/pull/10309)
* [`items_after_statements`]: `#[allow(items_after_statements)]` now works on items
  [#10542](https://github.com/rust-lang/rust-clippy/pull/10542)
* [`significant_drop_tightening`]: Was optimized
  [#10533](https://github.com/rust-lang/rust-clippy/pull/10533)

### False Positive Fixes

* [`single_component_path_imports`]: No longer lints if the import is used relative to `self`
  [#10566](https://github.com/rust-lang/rust-clippy/pull/10566)
* [`derivable_impls`]: No longer suggests deriving `Default` on generics with implicit arguments
  [#10399](https://github.com/rust-lang/rust-clippy/pull/10399)
* [`let_unit_value`]: No longer lints if the expression contains an `await`
  [#10439](https://github.com/rust-lang/rust-clippy/pull/10439)
* [`double_must_use`]: Now ignores `async` functions
  [#10589](https://github.com/rust-lang/rust-clippy/pull/10589)
* [`manual_clamp`]: No longer lints in constant context
  [#10479](https://github.com/rust-lang/rust-clippy/pull/10479)
* [`almost_swapped`]: Now ignores external macros
  [#10502](https://github.com/rust-lang/rust-clippy/pull/10502)
* [`nonminimal_bool`]: Now ignores macros
  [#10527](https://github.com/rust-lang/rust-clippy/pull/10527)
* [`needless_return`]: No longer lints match statements with incompatible branches
  [#10593](https://github.com/rust-lang/rust-clippy/pull/10593)
* [`use_self`]: Do not suggest using `Self` in const generic parameters
  [#10375](https://github.com/rust-lang/rust-clippy/pull/10375)
* [`mem_replace_option_with_none`]: No longer lints on field expressions
  [#10594](https://github.com/rust-lang/rust-clippy/pull/10594)
* [`items_after_statements`]: No longer lints on items from macros
  [#10542](https://github.com/rust-lang/rust-clippy/pull/10542)
* [`print_literal`], [`write_literal`]: No longer lint strings coming from the `file!()` macro
  [#10573](https://github.com/rust-lang/rust-clippy/pull/10573)
* [`uninit_vec`], [`uninit_assumed_init`]: Now check the types inside arrays and tuples
  [#10553](https://github.com/rust-lang/rust-clippy/pull/10553)
* [`almost_swapped`]: No longer lints if a variable is assigned to itself
  [#10499](https://github.com/rust-lang/rust-clippy/pull/10499)
* [`missing_docs_in_private_items`]: No longer lints on public items
  [#10324](https://github.com/rust-lang/rust-clippy/pull/10324)

### Suggestion Fixes/Improvements

* [`extra_unused_type_parameters`]: The suggestion is now machine applicable
  [#10536](https://github.com/rust-lang/rust-clippy/pull/10536)
* [`match_single_binding`]: Now adds a semicolon after the suggestion
  [#10470](https://github.com/rust-lang/rust-clippy/pull/10470)
* [`missing_const_for_fn`]: Now includes a note if the change could break compatibility
  [#10618](https://github.com/rust-lang/rust-clippy/pull/10618)
* [`cast_possible_truncation`]: Corrected suggestion for float and wildcard casts
  [#10496](https://github.com/rust-lang/rust-clippy/pull/10496)
* [`transmutes_expressible_as_ptr_casts`]: The suggestion now includes parentheses when they are required
  [#10454](https://github.com/rust-lang/rust-clippy/pull/10454)

### ICE Fixes

* [`needless_borrow`]: No longer panics on ambiguous projections
  [#10403](https://github.com/rust-lang/rust-clippy/pull/10403)
* [`multiple_unsafe_ops_per_block`]: Fix ICE when calling a function-like object in an unsafe block
  [#10405](https://github.com/rust-lang/rust-clippy/pull/10405)

### Others

* `clippy-driver` now searches parent directories for `clippy.toml` files
  [#10592](https://github.com/rust-lang/rust-clippy/pull/10592)
* Fixed a deserialization error for the `array-size-threshold` config value
  [#10423](https://github.com/rust-lang/rust-clippy/pull/10423)

## Rust 1.69

Released 2023-04-20

[View all 72 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2023-01-13T06%3A12%3A46Z..2023-02-25T23%3A48%3A10Z+base%3Amaster)

### New Lints

* [`no_mangle_with_rust_abi`]
  [#10369](https://github.com/rust-lang/rust-clippy/pull/10369)
* [`significant_drop_tightening`]
  [#10163](https://github.com/rust-lang/rust-clippy/pull/10163)
* [`suspicious_command_arg_space`]
  [#10317](https://github.com/rust-lang/rust-clippy/pull/10317)
* [`let_underscore_untyped`]
  [#10356](https://github.com/rust-lang/rust-clippy/pull/10356)
* [`question_mark_used`]
  [#10342](https://github.com/rust-lang/rust-clippy/pull/10342)
* [`extra_unused_type_parameters`]
  [#10028](https://github.com/rust-lang/rust-clippy/pull/10028)
* [`impl_trait_in_params`]
  [10197](https://github.com/rust-lang/rust-clippy/pull/10197)
* [`transmute_int_to_non_zero`]
  [#10360](https://github.com/rust-lang/rust-clippy/pull/10360)
* [`multiple_unsafe_ops_per_block`]
  [#10206](https://github.com/rust-lang/rust-clippy/pull/10206)

### Moves and Deprecations

* Moved [`uninlined_format_args`] to `pedantic` (Now allow-by-default)
  [#10265](https://github.com/rust-lang/rust-clippy/pull/10265)
* Moved [`unchecked_duration_subtraction`] to `pedantic` (Now allow-by-default)
  [#10194](https://github.com/rust-lang/rust-clippy/pull/10194)

### Enhancements

* [`arithmetic_side_effects`]: No longer lints if safe constant values are used.
  [#10310](https://github.com/rust-lang/rust-clippy/pull/10310)
* [`needless_lifetimes`]: Now works in local macros
  [#10257](https://github.com/rust-lang/rust-clippy/pull/10257)
* [`unused_io_amount`]: Now detects usages of `is_ok` and `is_err`
  [#10225](https://github.com/rust-lang/rust-clippy/pull/10225)
* [`missing_docs_in_private_items`]: Added new configuration `missing-docs-in-crate-items` to lint
  on items visible within the current crate. For example, `pub(crate)` items.
  [#10303](https://github.com/rust-lang/rust-clippy/pull/10303)
* [`almost_swapped`]: Now detects almost swaps using `let` statements
  [#10177](https://github.com/rust-lang/rust-clippy/pull/10177)
* [`wildcard_enum_match_arm`]: Now lints missing private variants, for local enums
  [#10250](https://github.com/rust-lang/rust-clippy/pull/10250)

### False Positive Fixes

* [`explicit_auto_deref`]: Now considers projections when determining if auto deref is applicable
  [#10386](https://github.com/rust-lang/rust-clippy/pull/10386)
* [`manual_let_else`]: Now considers side effects of branches before linting
  [#10336](https://github.com/rust-lang/rust-clippy/pull/10336)
* [`uninlined_format_args`]: No longer lints for arguments with generic parameters
  [#10343](https://github.com/rust-lang/rust-clippy/pull/10343)
* [`needless_lifetimes`]: No longer lints signatures in macros if the lifetime is a metavariable
  [#10380](https://github.com/rust-lang/rust-clippy/pull/10380)
* [`len_without_is_empty`]: No longer lints if `len` as a non-default signature
  [#10255](https://github.com/rust-lang/rust-clippy/pull/10255)
* [`unusual_byte_groupings`]: Relaxed the required restrictions for specific sizes to reduce false
  positives
  [#10353](https://github.com/rust-lang/rust-clippy/pull/10353)
* [`manual_let_else`]: No longer lints `if-else` blocks if they can divergent
  [#10332](https://github.com/rust-lang/rust-clippy/pull/10332)
* [`expect_used`], [`unwrap_used`], [`dbg_macro`], [`print_stdout`], [`print_stderr`]: No longer lint
  in test functions if `allow-expect-in-tests` is set
  [#10391](https://github.com/rust-lang/rust-clippy/pull/10391)
* [`unnecessary_safety_comment`]: No longer lints code inside macros
  [#10106](https://github.com/rust-lang/rust-clippy/pull/10106)
* [`never_loop`]: No longer lints statements following break statements for outer blocks.
  [#10311](https://github.com/rust-lang/rust-clippy/pull/10311)

### Suggestion Fixes/Improvements

* [`box_default`]: The suggestion now includes the type for trait objects when needed
  [#10382](https://github.com/rust-lang/rust-clippy/pull/10382)
* [`cast_possible_truncation`]: Now suggests using `try_from` or allowing the lint
  [#10038](https://github.com/rust-lang/rust-clippy/pull/10038)
* [`invalid_regex`]: Regex errors for non-literals or regular strings containing escape sequences will
  now show the complete error
  [#10231](https://github.com/rust-lang/rust-clippy/pull/10231)
* [`transmutes_expressible_as_ptr_casts`]: The suggestion now works if the base type is borrowed
  [#10193](https://github.com/rust-lang/rust-clippy/pull/10193)
* [`needless_return`]: Now removes all semicolons on the same line
  [#10187](https://github.com/rust-lang/rust-clippy/pull/10187)
* [`suspicious_to_owned`]: The suggestion now shows all options clearly
  [#10295](https://github.com/rust-lang/rust-clippy/pull/10295)
* [`bytes_nth`]: Now suggests the correct replacement based on the context
  [#10361](https://github.com/rust-lang/rust-clippy/pull/10361)
* [`bool_assert_comparison`]: The suggestion is now machine applicable
  [#10218](https://github.com/rust-lang/rust-clippy/pull/10218)
* [`cast_possible_truncation`]: Corrected the lint name in the help message
  [#10330](https://github.com/rust-lang/rust-clippy/pull/10330)
* [`needless_return`]: The suggestion now works on if sequences
  [#10345](https://github.com/rust-lang/rust-clippy/pull/10345)
* [`needless_lifetimes`]: The suggestion is now machine applicable
  [#10222](https://github.com/rust-lang/rust-clippy/pull/10222)
* [`map_entry`]: The suggestion no longer expands macros
  [#10346](https://github.com/rust-lang/rust-clippy/pull/10346)

### ICE Fixes

* [`needless_pass_by_value`]: Fixed an ICE caused by how late bounds were handled
  [#10328](https://github.com/rust-lang/rust-clippy/pull/10328)
* [`needless_borrow`]: No longer panics on ambiguous projections
  [#10403](https://github.com/rust-lang/rust-clippy/pull/10403)

### Documentation Improvements

* All configurations are now documented in the Clippy Book
  [#10199](https://github.com/rust-lang/rust-clippy/pull/10199)

## Rust 1.68

Released 2023-03-09

[View all 76 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2022-12-01T20%3A40%3A04Z..2023-01-12T18%3A58%3A59Z+base%3Amaster)

### New Lints

* [`permissions_set_readonly_false`]
  [#10063](https://github.com/rust-lang/rust-clippy/pull/10063)
* [`almost_complete_range`]
  [#10043](https://github.com/rust-lang/rust-clippy/pull/10043)
* [`size_of_ref`]
  [#10098](https://github.com/rust-lang/rust-clippy/pull/10098)
* [`semicolon_outside_block`]
  [#9826](https://github.com/rust-lang/rust-clippy/pull/9826)
* [`semicolon_inside_block`]
  [#9826](https://github.com/rust-lang/rust-clippy/pull/9826)
* [`transmute_null_to_fn`]
  [#10099](https://github.com/rust-lang/rust-clippy/pull/10099)
* [`fn_null_check`]
  [#10099](https://github.com/rust-lang/rust-clippy/pull/10099)

### Moves and Deprecations

* Moved [`manual_clamp`] to `nursery` (Now allow-by-default)
  [#10101](https://github.com/rust-lang/rust-clippy/pull/10101)
* Moved [`mutex_atomic`] to `restriction`
  [#10115](https://github.com/rust-lang/rust-clippy/pull/10115)
* Renamed `derive_hash_xor_eq` to [`derived_hash_with_manual_eq`]
  [#10184](https://github.com/rust-lang/rust-clippy/pull/10184)

### Enhancements

* [`collapsible_str_replace`]: Now takes MSRV into consideration. The minimal version is 1.58
  [#10047](https://github.com/rust-lang/rust-clippy/pull/10047)
* [`unused_self`]: No longer lints, if the method body contains a `todo!()` call
  [#10166](https://github.com/rust-lang/rust-clippy/pull/10166)
* [`derivable_impls`]: Now suggests deriving `Default` for enums with default unit variants
  [#10161](https://github.com/rust-lang/rust-clippy/pull/10161)
* [`arithmetic_side_effects`]: Added two new config values
  `arithmetic-side-effects-allowed-binary` and `arithmetic-side-effects-allowed-unary`
  to allow operation on user types
  [#9840](https://github.com/rust-lang/rust-clippy/pull/9840)
* [`large_const_arrays`], [`large_stack_arrays`]: avoid integer overflow when calculating
  total array size
  [#10103](https://github.com/rust-lang/rust-clippy/pull/10103)
* [`indexing_slicing`]: add new config `suppress-restriction-lint-in-const` to enable
  restriction lints, even if the suggestion might not be applicable
  [#9920](https://github.com/rust-lang/rust-clippy/pull/9920)
* [`needless_borrow`], [`redundant_clone`]: Now track references better and detect more cases
  [#9701](https://github.com/rust-lang/rust-clippy/pull/9701)
* [`derived_hash_with_manual_eq`]: Now allows `#[derive(PartialEq)]` with custom `Hash`
  implementations
  [#10184](https://github.com/rust-lang/rust-clippy/pull/10184)
* [`manual_is_ascii_check`]: Now detects ranges with `.contains()` calls
  [#10053](https://github.com/rust-lang/rust-clippy/pull/10053)
* [`transmuting_null`]: Now detects `const` pointers to all types
  [#10099](https://github.com/rust-lang/rust-clippy/pull/10099)
* [`needless_return`]: Now detects more cases for returns of owned values
  [#10110](https://github.com/rust-lang/rust-clippy/pull/10110)

### False Positive Fixes

* [`field_reassign_with_default`]: No longer lints cases, where values are initializes from
  closures capturing struct values
  [#10143](https://github.com/rust-lang/rust-clippy/pull/10143)
* [`seek_to_start_instead_of_rewind`]: No longer lints, if the return of `seek` is used.
  [#10096](https://github.com/rust-lang/rust-clippy/pull/10096)
* [`manual_filter`]: Now ignores if expressions where the else branch has side effects or
  doesn't return `None`
  [#10091](https://github.com/rust-lang/rust-clippy/pull/10091)
* [`implicit_clone`]: No longer lints if the type doesn't implement clone
  [#10022](https://github.com/rust-lang/rust-clippy/pull/10022)
* [`match_wildcard_for_single_variants`]: No longer lints on wildcards with a guard
  [#10056](https://github.com/rust-lang/rust-clippy/pull/10056)
* [`drop_ref`]: No longer lints idiomatic expression in `match` arms
  [#10142](https://github.com/rust-lang/rust-clippy/pull/10142)
* [`arithmetic_side_effects`]: No longer lints on corner cases with negative number literals
  [#9867](https://github.com/rust-lang/rust-clippy/pull/9867)
* [`string_lit_as_bytes`]: No longer lints in scrutinies of `match` statements
  [#10012](https://github.com/rust-lang/rust-clippy/pull/10012)
* [`manual_assert`]: No longer lints in `else if` statements
  [#10013](https://github.com/rust-lang/rust-clippy/pull/10013)
* [`needless_return`]: don't lint when using `do yeet`
  [#10109](https://github.com/rust-lang/rust-clippy/pull/10109)
* All lints: No longer lint in enum discriminant values when the suggestion won't work in a
  const context
  [#10008](https://github.com/rust-lang/rust-clippy/pull/10008)
* [`single_element_loop`]: No longer lints, if the loop contains a `break` or `continue`
  [#10162](https://github.com/rust-lang/rust-clippy/pull/10162)
* [`uninlined_format_args`]: No longer suggests inlining arguments in `assert!` and
  `debug_assert!` macros before 2021 edition
  [#10055](https://github.com/rust-lang/rust-clippy/pull/10055)
* [`explicit_counter_loop`]: No longer ignores counter changes after `continue` expressions
  [#10094](https://github.com/rust-lang/rust-clippy/pull/10094)
* [`from_over_into`]: No longer lints on opaque types
  [#9982](https://github.com/rust-lang/rust-clippy/pull/9982)
* [`expl_impl_clone_on_copy`]: No longer lints on `#[repr(packed)]` structs with generic
  parameters
  [#10189](https://github.com/rust-lang/rust-clippy/pull/10189)

### Suggestion Fixes/Improvements

* [`zero_ptr`]: Now suggests `core::` paths for `no_std` crates
  [#10023](https://github.com/rust-lang/rust-clippy/pull/10023)
* [`useless_conversion`]: Now suggests removing calls to `into_iter()` on an expression
  implementing `Iterator`
  [#10020](https://github.com/rust-lang/rust-clippy/pull/10020)
* [`box_default`]: The suggestion now uses short paths
  [#10153](https://github.com/rust-lang/rust-clippy/pull/10153)
* [`default_trait_access`], [`clone_on_copy`]: The suggestion now uses short paths
  [#10160](https://github.com/rust-lang/rust-clippy/pull/10160)
* [`comparison_to_empty`]: The suggestion now removes unused deref operations
  [#9962](https://github.com/rust-lang/rust-clippy/pull/9962)
* [`manual_let_else`]: Suggestions for or-patterns now include required brackets.
  [#9966](https://github.com/rust-lang/rust-clippy/pull/9966)
* [`match_single_binding`]: suggestion no longer introduces unneeded semicolons
  [#10060](https://github.com/rust-lang/rust-clippy/pull/10060)
* [`case_sensitive_file_extension_comparisons`]: Now displays a suggestion with `Path`
  [#10107](https://github.com/rust-lang/rust-clippy/pull/10107)
* [`empty_structs_with_brackets`]: The suggestion is no longer machine applicable, to avoid
  errors when accessing struct fields
  [#10141](https://github.com/rust-lang/rust-clippy/pull/10141)
* [`identity_op`]: Removes borrows in the suggestion when needed
  [#10004](https://github.com/rust-lang/rust-clippy/pull/10004)
* [`suboptimal_flops`]: The suggestion now includes parentheses when required
  [#10113](https://github.com/rust-lang/rust-clippy/pull/10113)
* [`iter_kv_map`]: Now handles `mut` and reference annotations in the suggestion
  [#10159](https://github.com/rust-lang/rust-clippy/pull/10159)
* [`redundant_static_lifetimes`]: The suggestion no longer removes `mut` from references
  [#10006](https://github.com/rust-lang/rust-clippy/pull/10006)

### ICE Fixes

* [`new_ret_no_self`]: Now avoids a stack overflow for `impl Trait` types
  [#10086](https://github.com/rust-lang/rust-clippy/pull/10086)
* [`unnecessary_to_owned`]: Now handles compiler generated notes better
  [#10027](https://github.com/rust-lang/rust-clippy/pull/10027)

### Others

* `SYSROOT` and `--sysroot` can now be set at the same time
  [#10149](https://github.com/rust-lang/rust-clippy/pull/10149)
* Fix error when providing an `array-size-threshold` in `clippy.toml`
  [#10423](https://github.com/rust-lang/rust-clippy/pull/10423)

## Rust 1.67

Released 2023-01-26

[View all 104 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2022-10-23T13%3A35%3A19Z..2022-12-01T13%3A34%3A39Z+base%3Amaster)

### New Lints

* [`seek_from_current`]
  [#9681](https://github.com/rust-lang/rust-clippy/pull/9681)
* [`from_raw_with_void_ptr`]
  [#9690](https://github.com/rust-lang/rust-clippy/pull/9690)
* [`misnamed_getters`]
  [#9770](https://github.com/rust-lang/rust-clippy/pull/9770)
* [`seek_to_start_instead_of_rewind`]
  [#9667](https://github.com/rust-lang/rust-clippy/pull/9667)
* [`suspicious_xor_used_as_pow`]
  [#9506](https://github.com/rust-lang/rust-clippy/pull/9506)
* [`unnecessary_safety_doc`]
  [#9822](https://github.com/rust-lang/rust-clippy/pull/9822)
* [`unchecked_duration_subtraction`]
  [#9570](https://github.com/rust-lang/rust-clippy/pull/9570)
* [`manual_is_ascii_check`]
  [#9765](https://github.com/rust-lang/rust-clippy/pull/9765)
* [`unnecessary_safety_comment`]
  [#9851](https://github.com/rust-lang/rust-clippy/pull/9851)
* [`let_underscore_future`]
  [#9760](https://github.com/rust-lang/rust-clippy/pull/9760)
* [`manual_let_else`]
  [#8437](https://github.com/rust-lang/rust-clippy/pull/8437)

### Moves and Deprecations

* Moved [`needless_collect`] to `nursery` (Now allow-by-default)
  [#9705](https://github.com/rust-lang/rust-clippy/pull/9705)
* Moved [`or_fun_call`] to `nursery` (Now allow-by-default)
  [#9829](https://github.com/rust-lang/rust-clippy/pull/9829)
* Uplifted [`let_underscore_lock`] into rustc
  [#9697](https://github.com/rust-lang/rust-clippy/pull/9697)
* Uplifted [`let_underscore_drop`] into rustc
  [#9697](https://github.com/rust-lang/rust-clippy/pull/9697)
* Moved [`bool_to_int_with_if`] to `pedantic` (Now allow-by-default)
  [#9830](https://github.com/rust-lang/rust-clippy/pull/9830)
* Move `index_refutable_slice` to `pedantic` (Now warn-by-default)
  [#9975](https://github.com/rust-lang/rust-clippy/pull/9975)
* Moved [`manual_clamp`] to `nursery` (Now allow-by-default)
  [#10101](https://github.com/rust-lang/rust-clippy/pull/10101)

### Enhancements

* The scope of `#![clippy::msrv]` is now tracked correctly
  [#9924](https://github.com/rust-lang/rust-clippy/pull/9924)
* `#[clippy::msrv]` can now be used as an outer attribute
  [#9860](https://github.com/rust-lang/rust-clippy/pull/9860)
* Clippy will now avoid Cargo's cache, if `Cargo.toml` or `clippy.toml` have changed
  [#9707](https://github.com/rust-lang/rust-clippy/pull/9707)
* [`uninlined_format_args`]: Added a new config `allow-mixed-uninlined-format-args` to allow the
  lint, if only some arguments can be inlined
  [#9865](https://github.com/rust-lang/rust-clippy/pull/9865)
* [`needless_lifetimes`]: Now provides suggests for individual lifetimes
  [#9743](https://github.com/rust-lang/rust-clippy/pull/9743)
* [`needless_collect`]: Now detects needless `is_empty` and `contains` calls
  [#8744](https://github.com/rust-lang/rust-clippy/pull/8744)
* [`blanket_clippy_restriction_lints`]: Now lints, if `clippy::restriction` is enabled via the
  command line arguments
  [#9755](https://github.com/rust-lang/rust-clippy/pull/9755)
* [`mutable_key_type`]: Now has the `ignore-interior-mutability` configuration, to add types which
  should be ignored by the lint
  [#9692](https://github.com/rust-lang/rust-clippy/pull/9692)
* [`uninlined_format_args`]: Now works for multiline `format!` expressions
  [#9945](https://github.com/rust-lang/rust-clippy/pull/9945)
* [`cognitive_complexity`]: Now works for async functions
  [#9828](https://github.com/rust-lang/rust-clippy/pull/9828)
  [#9836](https://github.com/rust-lang/rust-clippy/pull/9836)
* [`vec_box`]: Now avoids an off-by-one error when using the `vec-box-size-threshold` configuration
  [#9848](https://github.com/rust-lang/rust-clippy/pull/9848)
* [`never_loop`]: Now correctly handles breaks in nested labeled blocks
  [#9858](https://github.com/rust-lang/rust-clippy/pull/9858)
  [#9837](https://github.com/rust-lang/rust-clippy/pull/9837)
* [`disallowed_methods`], [`disallowed_types`], [`disallowed_macros`]: Now correctly resolve
  paths, if a crate is used multiple times with different versions
  [#9800](https://github.com/rust-lang/rust-clippy/pull/9800)
* [`disallowed_methods`]: Can now be used for local methods
  [#9800](https://github.com/rust-lang/rust-clippy/pull/9800)
* [`print_stdout`], [`print_stderr`]: Can now be enabled in test with the `allow-print-in-tests`
  config value
  [#9797](https://github.com/rust-lang/rust-clippy/pull/9797)
* [`from_raw_with_void_ptr`]: Now works for `Rc`, `Arc`, `alloc::rc::Weak` and
  `alloc::sync::Weak` types.
  [#9700](https://github.com/rust-lang/rust-clippy/pull/9700)
* [`needless_borrowed_reference`]: Now works for struct and tuple patterns with wildcards
  [#9855](https://github.com/rust-lang/rust-clippy/pull/9855)
* [`or_fun_call`]: Now supports `map_or` methods
  [#9689](https://github.com/rust-lang/rust-clippy/pull/9689)
* [`unwrap_used`], [`expect_used`]: No longer lints in test code
  [#9686](https://github.com/rust-lang/rust-clippy/pull/9686)
* [`fn_params_excessive_bools`]: Is now emitted with the lint level at the linted function
  [#9698](https://github.com/rust-lang/rust-clippy/pull/9698)

### False Positive Fixes

* [`new_ret_no_self`]: No longer lints when `impl Trait<Self>` is returned
  [#9733](https://github.com/rust-lang/rust-clippy/pull/9733)
* [`unnecessary_lazy_evaluations`]: No longer lints, if the type has a significant drop
  [#9750](https://github.com/rust-lang/rust-clippy/pull/9750)
* [`option_if_let_else`]: No longer lints, if any arm has guard
  [#9747](https://github.com/rust-lang/rust-clippy/pull/9747)
* [`explicit_auto_deref`]: No longer lints, if the target type is a projection with generic
  arguments
  [#9813](https://github.com/rust-lang/rust-clippy/pull/9813)
* [`unnecessary_to_owned`]: No longer lints, if the suggestion effects types
  [#9796](https://github.com/rust-lang/rust-clippy/pull/9796)
* [`needless_borrow`]: No longer lints, if the suggestion is affected by `Deref`
  [#9674](https://github.com/rust-lang/rust-clippy/pull/9674)
* [`unused_unit`]: No longer lints, if lifetimes are bound to the return type
  [#9849](https://github.com/rust-lang/rust-clippy/pull/9849)
* [`mut_mut`]: No longer lints cases with unsized mutable references
  [#9835](https://github.com/rust-lang/rust-clippy/pull/9835)
* [`bool_to_int_with_if`]: No longer lints in const context
  [#9738](https://github.com/rust-lang/rust-clippy/pull/9738)
* [`use_self`]: No longer lints in macros
  [#9704](https://github.com/rust-lang/rust-clippy/pull/9704)
* [`unnecessary_operation`]: No longer lints, if multiple macros are involved
  [#9981](https://github.com/rust-lang/rust-clippy/pull/9981)
* [`allow_attributes_without_reason`]: No longer lints inside external macros
  [#9630](https://github.com/rust-lang/rust-clippy/pull/9630)
* [`question_mark`]: No longer lints for `if let Err()` with an `else` branch
  [#9722](https://github.com/rust-lang/rust-clippy/pull/9722)
* [`unnecessary_cast`]: No longer lints if the identifier and cast originate from different macros
  [#9980](https://github.com/rust-lang/rust-clippy/pull/9980)
* [`arithmetic_side_effects`]: Now detects operations with associated constants
  [#9592](https://github.com/rust-lang/rust-clippy/pull/9592)
* [`explicit_auto_deref`]: No longer lints, if the initial value is not a reference or reference
  receiver
  [#9997](https://github.com/rust-lang/rust-clippy/pull/9997)
* [`module_name_repetitions`], [`single_component_path_imports`]: Now handle `#[allow]`
  attributes correctly
  [#9879](https://github.com/rust-lang/rust-clippy/pull/9879)
* [`bool_to_int_with_if`]: No longer lints `if let` statements
  [#9714](https://github.com/rust-lang/rust-clippy/pull/9714)
* [`needless_borrow`]: No longer lints, `if`-`else`-statements that require the borrow
  [#9791](https://github.com/rust-lang/rust-clippy/pull/9791)
* [`needless_borrow`]: No longer lints borrows, if moves were illegal
  [#9711](https://github.com/rust-lang/rust-clippy/pull/9711)
* [`manual_swap`]: No longer lints in const context
  [#9871](https://github.com/rust-lang/rust-clippy/pull/9871)

### Suggestion Fixes/Improvements

* [`missing_safety_doc`], [`missing_errors_doc`], [`missing_panics_doc`]: No longer show the
  entire item in the lint emission.
  [#9772](https://github.com/rust-lang/rust-clippy/pull/9772)
* [`needless_lifetimes`]: Only suggests `'_` when it's applicable
  [#9743](https://github.com/rust-lang/rust-clippy/pull/9743)
* [`use_self`]: Now suggests full paths correctly
  [#9726](https://github.com/rust-lang/rust-clippy/pull/9726)
* [`redundant_closure_call`]: Now correctly deals with macros during suggestion creation
  [#9987](https://github.com/rust-lang/rust-clippy/pull/9987)
* [`unnecessary_cast`]: Suggestions now correctly deal with references
  [#9996](https://github.com/rust-lang/rust-clippy/pull/9996)
* [`unnecessary_join`]: Suggestions now correctly use [turbofish] operators
  [#9779](https://github.com/rust-lang/rust-clippy/pull/9779)
* [`equatable_if_let`]: Can now suggest `matches!` replacements
  [#9368](https://github.com/rust-lang/rust-clippy/pull/9368)
* [`string_extend_chars`]: Suggestions now correctly work for `str` slices
  [#9741](https://github.com/rust-lang/rust-clippy/pull/9741)
* [`redundant_closure_for_method_calls`]: Suggestions now include angle brackets and generic
  arguments if needed
  [#9745](https://github.com/rust-lang/rust-clippy/pull/9745)
* [`manual_let_else`]: Suggestions no longer expand macro calls
  [#9943](https://github.com/rust-lang/rust-clippy/pull/9943)
* [`infallible_destructuring_match`]: Suggestions now preserve references
  [#9850](https://github.com/rust-lang/rust-clippy/pull/9850)
* [`result_large_err`]: The error now shows the largest enum variant
  [#9662](https://github.com/rust-lang/rust-clippy/pull/9662)
* [`needless_return`]: Suggestions are now formatted better
  [#9967](https://github.com/rust-lang/rust-clippy/pull/9967)
* [`unused_rounding`]: The suggestion now preserves the original float literal notation
  [#9870](https://github.com/rust-lang/rust-clippy/pull/9870)

[turbofish]: https://turbo.fish/::%3CClippy%3E

### ICE Fixes

* [`result_large_err`]: Fixed ICE for empty enums
  [#10007](https://github.com/rust-lang/rust-clippy/pull/10007)
* [`redundant_allocation`]: Fixed ICE for types with bounded variables
  [#9773](https://github.com/rust-lang/rust-clippy/pull/9773)
* [`unused_rounding`]: Fixed ICE, if `_` was used as a separator
  [#10001](https://github.com/rust-lang/rust-clippy/pull/10001)

## Rust 1.66

Released 2022-12-15

[View all 116 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2022-09-09T17%3A32%3A39Z..2022-10-23T11%3A27%3A24Z+base%3Amaster)

### New Lints

* [`manual_clamp`]
  [#9484](https://github.com/rust-lang/rust-clippy/pull/9484)
* [`missing_trait_methods`]
  [#9670](https://github.com/rust-lang/rust-clippy/pull/9670)
* [`unused_format_specs`]
  [#9637](https://github.com/rust-lang/rust-clippy/pull/9637)
* [`iter_kv_map`]
  [#9409](https://github.com/rust-lang/rust-clippy/pull/9409)
* [`manual_filter`]
  [#9451](https://github.com/rust-lang/rust-clippy/pull/9451)
* [`box_default`]
  [#9511](https://github.com/rust-lang/rust-clippy/pull/9511)
* [`implicit_saturating_add`]
  [#9549](https://github.com/rust-lang/rust-clippy/pull/9549)
* [`as_ptr_cast_mut`]
  [#9572](https://github.com/rust-lang/rust-clippy/pull/9572)
* [`disallowed_macros`]
  [#9495](https://github.com/rust-lang/rust-clippy/pull/9495)
* [`partial_pub_fields`]
  [#9658](https://github.com/rust-lang/rust-clippy/pull/9658)
* [`uninlined_format_args`]
  [#9233](https://github.com/rust-lang/rust-clippy/pull/9233)
* [`cast_nan_to_int`]
  [#9617](https://github.com/rust-lang/rust-clippy/pull/9617)

### Moves and Deprecations

* `positional_named_format_parameters` was uplifted to rustc under the new name
  `named_arguments_used_positionally`
  [#8518](https://github.com/rust-lang/rust-clippy/pull/8518)
* Moved [`implicit_saturating_sub`] to `style` (Now warn-by-default)
  [#9584](https://github.com/rust-lang/rust-clippy/pull/9584)
* Moved `derive_partial_eq_without_eq` to `nursery` (now allow-by-default)
  [#9536](https://github.com/rust-lang/rust-clippy/pull/9536)

### Enhancements

* [`nonstandard_macro_braces`]: Now includes `matches!()` in the default lint config
  [#9471](https://github.com/rust-lang/rust-clippy/pull/9471)
* [`suboptimal_flops`]: Now supports multiplication and subtraction operations
  [#9581](https://github.com/rust-lang/rust-clippy/pull/9581)
* [`arithmetic_side_effects`]: Now detects cases with literals behind references
  [#9587](https://github.com/rust-lang/rust-clippy/pull/9587)
* [`upper_case_acronyms`]: Now also checks enum names
  [#9580](https://github.com/rust-lang/rust-clippy/pull/9580)
* [`needless_borrowed_reference`]: Now lints nested patterns
  [#9573](https://github.com/rust-lang/rust-clippy/pull/9573)
* [`unnecessary_cast`]: Now works for non-trivial non-literal expressions
  [#9576](https://github.com/rust-lang/rust-clippy/pull/9576)
* [`arithmetic_side_effects`]: Now detects operations with custom types
  [#9559](https://github.com/rust-lang/rust-clippy/pull/9559)
* [`disallowed_methods`], [`disallowed_types`]: Not correctly lints types, functions and macros
  with the same path
  [#9495](https://github.com/rust-lang/rust-clippy/pull/9495)
* [`self_named_module_files`], [`mod_module_files`]: Now take remapped path prefixes into account
  [#9475](https://github.com/rust-lang/rust-clippy/pull/9475)
* [`bool_to_int_with_if`]: Now detects the inverse if case
  [#9476](https://github.com/rust-lang/rust-clippy/pull/9476)

### False Positive Fixes

* [`arithmetic_side_effects`]: Now allows operations that can't overflow
  [#9474](https://github.com/rust-lang/rust-clippy/pull/9474)
* [`unnecessary_lazy_evaluations`]: No longer lints in external macros
  [#9486](https://github.com/rust-lang/rust-clippy/pull/9486)
* [`needless_borrow`], [`explicit_auto_deref`]: No longer lint on unions that require the reference
  [#9490](https://github.com/rust-lang/rust-clippy/pull/9490)
* [`almost_complete_letter_range`]: No longer lints in external macros
  [#9467](https://github.com/rust-lang/rust-clippy/pull/9467)
* [`drop_copy`]: No longer lints on idiomatic cases in match arms
  [#9491](https://github.com/rust-lang/rust-clippy/pull/9491)
* [`question_mark`]: No longer lints in const context
  [#9487](https://github.com/rust-lang/rust-clippy/pull/9487)
* [`collapsible_if`]: Suggestion now work in macros
  [#9410](https://github.com/rust-lang/rust-clippy/pull/9410)
* [`std_instead_of_core`]: No longer triggers on unstable modules
  [#9545](https://github.com/rust-lang/rust-clippy/pull/9545)
* [`unused_peekable`]: No longer lints, if the peak is done in a closure or function
  [#9465](https://github.com/rust-lang/rust-clippy/pull/9465)
* [`useless_attribute`]: No longer lints on `#[allow]` attributes for [`unsafe_removed_from_name`]
  [#9593](https://github.com/rust-lang/rust-clippy/pull/9593)
* [`unnecessary_lazy_evaluations`]: No longer suggest switching to early evaluation when type has
  custom `Drop` implementation
  [#9551](https://github.com/rust-lang/rust-clippy/pull/9551)
* [`unnecessary_cast`]: No longer lints on negative hexadecimal literals when cast as floats
  [#9609](https://github.com/rust-lang/rust-clippy/pull/9609)
* [`use_self`]: No longer lints in proc macros
  [#9454](https://github.com/rust-lang/rust-clippy/pull/9454)
* [`never_loop`]: Now takes `let ... else` statements into consideration.
  [#9496](https://github.com/rust-lang/rust-clippy/pull/9496)
* [`default_numeric_fallback`]: Now ignores constants
  [#9636](https://github.com/rust-lang/rust-clippy/pull/9636)
* [`uninit_vec`]: No longer lints `Vec::set_len(0)`
  [#9519](https://github.com/rust-lang/rust-clippy/pull/9519)
* [`arithmetic_side_effects`]: Now ignores references to integer types
  [#9507](https://github.com/rust-lang/rust-clippy/pull/9507)
* [`large_stack_arrays`]: No longer lints inside static items
  [#9466](https://github.com/rust-lang/rust-clippy/pull/9466)
* [`ref_option_ref`]: No longer lints if the inner reference is mutable
  [#9684](https://github.com/rust-lang/rust-clippy/pull/9684)
* [`ptr_arg`]: No longer lints if the argument is used as an incomplete trait object
  [#9645](https://github.com/rust-lang/rust-clippy/pull/9645)
* [`should_implement_trait`]: Now also works for `default` methods
  [#9546](https://github.com/rust-lang/rust-clippy/pull/9546)

### Suggestion Fixes/Improvements

* [`derivable_impls`]: The suggestion is now machine applicable
  [#9429](https://github.com/rust-lang/rust-clippy/pull/9429)
* [`match_single_binding`]: The suggestion now handles scrutinies with side effects better
  [#9601](https://github.com/rust-lang/rust-clippy/pull/9601)
* [`zero_prefixed_literal`]: Only suggests using octal numbers, if this is possible
  [#9652](https://github.com/rust-lang/rust-clippy/pull/9652)
* [`rc_buffer`]: The suggestion is no longer machine applicable to avoid semantic changes
  [#9633](https://github.com/rust-lang/rust-clippy/pull/9633)
* [`print_literal`], [`write_literal`], [`uninlined_format_args`]: The suggestion now ignores
  comments after the macro call.
  [#9586](https://github.com/rust-lang/rust-clippy/pull/9586)
* [`expect_fun_call`]:Improved the suggestion for `format!` calls with captured variables
  [#9586](https://github.com/rust-lang/rust-clippy/pull/9586)
* [`nonstandard_macro_braces`]: The suggestion is now machine applicable and will no longer
  replace brackets inside the macro argument.
  [#9499](https://github.com/rust-lang/rust-clippy/pull/9499)
* [`from_over_into`]: The suggestion is now a machine applicable and contains explanations
  [#9649](https://github.com/rust-lang/rust-clippy/pull/9649)
* [`needless_return`]: The automatic suggestion now removes all required semicolons
  [#9497](https://github.com/rust-lang/rust-clippy/pull/9497)
* [`to_string_in_format_args`]: The suggestion now keeps parenthesis around values
  [#9590](https://github.com/rust-lang/rust-clippy/pull/9590)
* [`manual_assert`]: The suggestion now preserves comments
  [#9479](https://github.com/rust-lang/rust-clippy/pull/9479)
* [`redundant_allocation`]: The suggestion applicability is now marked `MaybeIncorrect` to
  avoid semantic changes
  [#9634](https://github.com/rust-lang/rust-clippy/pull/9634)
* [`assertions_on_result_states`]: The suggestion has been corrected, for cases where the
  `assert!` is not in a statement.
  [#9453](https://github.com/rust-lang/rust-clippy/pull/9453)
* [`nonminimal_bool`]: The suggestion no longer expands macros
  [#9457](https://github.com/rust-lang/rust-clippy/pull/9457)
* [`collapsible_match`]: Now specifies field names, when a struct is destructed
  [#9685](https://github.com/rust-lang/rust-clippy/pull/9685)
* [`unnecessary_cast`]: The suggestion now adds parenthesis for negative numbers
  [#9577](https://github.com/rust-lang/rust-clippy/pull/9577)
* [`redundant_closure`]: The suggestion now works for `impl FnMut` arguments
  [#9556](https://github.com/rust-lang/rust-clippy/pull/9556)

### ICE Fixes

* [`unnecessary_to_owned`]: Avoid ICEs in favor of false negatives if information is missing
  [#9505](https://github.com/rust-lang/rust-clippy/pull/9505)
  [#10027](https://github.com/rust-lang/rust-clippy/pull/10027)
* [`manual_range_contains`]: No longer ICEs on values behind references
  [#9627](https://github.com/rust-lang/rust-clippy/pull/9627)
* [`needless_pass_by_value`]: No longer ICEs on unsized `dyn Fn` arguments
  [#9531](https://github.com/rust-lang/rust-clippy/pull/9531)
* `*_interior_mutable_const` lints: no longer ICE on const unions containing `!Freeze` types
  [#9539](https://github.com/rust-lang/rust-clippy/pull/9539)

### Others

* Released `rustc_tools_util` for version information on `Crates.io`. (Further adjustments will
  not be published as part of this changelog)

## Rust 1.65

Released 2022-11-03

[View all 86 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2022-07-29T01%3A09%3A31Z..2022-09-09T00%3A01%3A54Z+base%3Amaster)

### Important Changes

* Clippy now has an `--explain <LINT>` command to show the lint description in the console
  [#8952](https://github.com/rust-lang/rust-clippy/pull/8952)

### New Lints

* [`unused_peekable`]
  [#9258](https://github.com/rust-lang/rust-clippy/pull/9258)
* [`collapsible_str_replace`]
  [#9269](https://github.com/rust-lang/rust-clippy/pull/9269)
* [`manual_string_new`]
  [#9295](https://github.com/rust-lang/rust-clippy/pull/9295)
* [`iter_on_empty_collections`]
  [#9187](https://github.com/rust-lang/rust-clippy/pull/9187)
* [`iter_on_single_items`]
  [#9187](https://github.com/rust-lang/rust-clippy/pull/9187)
* [`bool_to_int_with_if`]
  [#9412](https://github.com/rust-lang/rust-clippy/pull/9412)
* [`multi_assignments`]
  [#9379](https://github.com/rust-lang/rust-clippy/pull/9379)
* [`result_large_err`]
  [#9373](https://github.com/rust-lang/rust-clippy/pull/9373)
* [`partialeq_to_none`]
  [#9288](https://github.com/rust-lang/rust-clippy/pull/9288)
* [`suspicious_to_owned`]
  [#8984](https://github.com/rust-lang/rust-clippy/pull/8984)
* [`cast_slice_from_raw_parts`]
  [#9247](https://github.com/rust-lang/rust-clippy/pull/9247)
* [`manual_instant_elapsed`]
  [#9264](https://github.com/rust-lang/rust-clippy/pull/9264)

### Moves and Deprecations

* Moved [`significant_drop_in_scrutinee`] to `nursery` (now allow-by-default)
  [#9302](https://github.com/rust-lang/rust-clippy/pull/9302)
* Rename `logic_bug` to [`overly_complex_bool_expr`]
  [#9306](https://github.com/rust-lang/rust-clippy/pull/9306)
* Rename `arithmetic` to [`arithmetic_side_effects`]
  [#9443](https://github.com/rust-lang/rust-clippy/pull/9443)
* Moved [`only_used_in_recursion`] to complexity (now warn-by-default)
  [#8804](https://github.com/rust-lang/rust-clippy/pull/8804)
* Moved [`assertions_on_result_states`] to restriction (now allow-by-default)
  [#9273](https://github.com/rust-lang/rust-clippy/pull/9273)
* Renamed `blacklisted_name` to [`disallowed_names`]
  [#8974](https://github.com/rust-lang/rust-clippy/pull/8974)

### Enhancements

* [`option_if_let_else`]: Now also checks for match expressions
  [#8696](https://github.com/rust-lang/rust-clippy/pull/8696)
* [`explicit_auto_deref`]: Now lints on implicit returns in closures
  [#9126](https://github.com/rust-lang/rust-clippy/pull/9126)
* [`needless_borrow`]: Now considers trait implementations
  [#9136](https://github.com/rust-lang/rust-clippy/pull/9136)
* [`suboptimal_flops`], [`imprecise_flops`]: Now lint on constant expressions
  [#9404](https://github.com/rust-lang/rust-clippy/pull/9404)
* [`if_let_mutex`]: Now detects mutex behind references and warns about deadlocks
  [#9318](https://github.com/rust-lang/rust-clippy/pull/9318)

### False Positive Fixes

* [`unit_arg`] [`default_trait_access`] [`missing_docs_in_private_items`]: No longer
  trigger in code generated from proc-macros
  [#8694](https://github.com/rust-lang/rust-clippy/pull/8694)
* [`unwrap_used`]: Now lints uses of `unwrap_err`
  [#9338](https://github.com/rust-lang/rust-clippy/pull/9338)
* [`expect_used`]: Now lints uses of `expect_err`
  [#9338](https://github.com/rust-lang/rust-clippy/pull/9338)
* [`transmute_undefined_repr`]: Now longer lints if the first field is compatible
  with the other type
  [#9287](https://github.com/rust-lang/rust-clippy/pull/9287)
* [`unnecessary_to_owned`]: No longer lints, if type change cased errors in
  the caller function
  [#9424](https://github.com/rust-lang/rust-clippy/pull/9424)
* [`match_like_matches_macro`]: No longer lints, if there are comments inside the
  match expression
  [#9276](https://github.com/rust-lang/rust-clippy/pull/9276)
* [`partialeq_to_none`]: No longer trigger in code generated from macros
  [#9389](https://github.com/rust-lang/rust-clippy/pull/9389)
* [`arithmetic_side_effects`]: No longer lints expressions that only use literals
  [#9365](https://github.com/rust-lang/rust-clippy/pull/9365)
* [`explicit_auto_deref`]: Now ignores references on block expressions when the type
  is `Sized`, on `dyn Trait` returns and when the suggestion is non-trivial
  [#9126](https://github.com/rust-lang/rust-clippy/pull/9126)
* [`trait_duplication_in_bounds`]: Now better tracks bounds to avoid false positives
  [#9167](https://github.com/rust-lang/rust-clippy/pull/9167)
* [`format_in_format_args`]: Now suggests cases where the result is formatted again
  [#9349](https://github.com/rust-lang/rust-clippy/pull/9349)
* [`only_used_in_recursion`]: No longer lints on function without recursions and
  takes external functions into account
  [#8804](https://github.com/rust-lang/rust-clippy/pull/8804)
* [`missing_const_for_fn`]: No longer lints in proc-macros
  [#9308](https://github.com/rust-lang/rust-clippy/pull/9308)
* [`non_ascii_literal`]: Allow non-ascii comments in tests and make sure `#[allow]`
  attributes work in tests
  [#9327](https://github.com/rust-lang/rust-clippy/pull/9327)
* [`question_mark`]: No longer lint `if let`s with subpatterns
  [#9348](https://github.com/rust-lang/rust-clippy/pull/9348)
* [`needless_collect`]: No longer lints in loops
  [#8992](https://github.com/rust-lang/rust-clippy/pull/8992)
* [`mut_mutex_lock`]: No longer lints if the mutex is behind an immutable reference
  [#9418](https://github.com/rust-lang/rust-clippy/pull/9418)
* [`needless_return`]: Now ignores returns with arguments
  [#9381](https://github.com/rust-lang/rust-clippy/pull/9381)
* [`range_plus_one`], [`range_minus_one`]: Now ignores code with macros
  [#9446](https://github.com/rust-lang/rust-clippy/pull/9446)
* [`assertions_on_result_states`]: No longer lints on the unit type
  [#9273](https://github.com/rust-lang/rust-clippy/pull/9273)

### Suggestion Fixes/Improvements

* [`unwrap_or_else_default`]: Now suggests `unwrap_or_default()` for empty strings
  [#9421](https://github.com/rust-lang/rust-clippy/pull/9421)
* [`if_then_some_else_none`]: Now also suggests `bool::then_some`
  [#9289](https://github.com/rust-lang/rust-clippy/pull/9289)
* [`redundant_closure_call`]: The suggestion now works for async closures
  [#9053](https://github.com/rust-lang/rust-clippy/pull/9053)
* [`suboptimal_flops`]: Now suggests parenthesis when they are required
  [#9394](https://github.com/rust-lang/rust-clippy/pull/9394)
* [`case_sensitive_file_extension_comparisons`]: Now suggests `map_or(..)` instead of `map(..).unwrap_or`
  [#9341](https://github.com/rust-lang/rust-clippy/pull/9341)
* Deprecated configuration values can now be updated automatically
  [#9252](https://github.com/rust-lang/rust-clippy/pull/9252)
* [`or_fun_call`]: Now suggest `Entry::or_default` for `Entry::or_insert(Default::default())`
  [#9342](https://github.com/rust-lang/rust-clippy/pull/9342)
* [`unwrap_used`]: Only suggests `expect` if [`expect_used`] is allowed
  [#9223](https://github.com/rust-lang/rust-clippy/pull/9223)

### ICE Fixes

* Fix ICE in [`useless_format`] for literals
  [#9406](https://github.com/rust-lang/rust-clippy/pull/9406)
* Fix infinite loop in [`vec_init_then_push`]
  [#9441](https://github.com/rust-lang/rust-clippy/pull/9441)
* Fix ICE when reading literals with weird proc-macro spans
  [#9303](https://github.com/rust-lang/rust-clippy/pull/9303)

## Rust 1.64

Released 2022-09-22

[View all 110 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2022-06-17T21%3A25%3A31Z..2022-07-28T17%3A11%3A18Z+base%3Amaster)

### New Lints

* [`arithmetic_side_effects`]
  [#9130](https://github.com/rust-lang/rust-clippy/pull/9130)
* [`invalid_utf8_in_unchecked`]
  [#9105](https://github.com/rust-lang/rust-clippy/pull/9105)
* [`assertions_on_result_states`]
  [#9225](https://github.com/rust-lang/rust-clippy/pull/9225)
* [`manual_find`]
  [#8649](https://github.com/rust-lang/rust-clippy/pull/8649)
* [`manual_retain`]
  [#8972](https://github.com/rust-lang/rust-clippy/pull/8972)
* [`default_instead_of_iter_empty`]
  [#8989](https://github.com/rust-lang/rust-clippy/pull/8989)
* [`manual_rem_euclid`]
  [#9031](https://github.com/rust-lang/rust-clippy/pull/9031)
* [`obfuscated_if_else`]
  [#9148](https://github.com/rust-lang/rust-clippy/pull/9148)
* [`std_instead_of_core`]
  [#9103](https://github.com/rust-lang/rust-clippy/pull/9103)
* [`std_instead_of_alloc`]
  [#9103](https://github.com/rust-lang/rust-clippy/pull/9103)
* [`alloc_instead_of_core`]
  [#9103](https://github.com/rust-lang/rust-clippy/pull/9103)
* [`explicit_auto_deref`]
  [#8355](https://github.com/rust-lang/rust-clippy/pull/8355)


### Moves and Deprecations

* Moved [`format_push_string`] to `restriction` (now allow-by-default)
  [#9161](https://github.com/rust-lang/rust-clippy/pull/9161)

### Enhancements

* [`significant_drop_in_scrutinee`]: Now gives more context in the lint message
  [#8981](https://github.com/rust-lang/rust-clippy/pull/8981)
* [`single_match`], [`single_match_else`]: Now catches more `Option` cases
  [#8985](https://github.com/rust-lang/rust-clippy/pull/8985)
* [`unused_async`]: Now works for async methods
  [#9025](https://github.com/rust-lang/rust-clippy/pull/9025)
* [`manual_filter_map`], [`manual_find_map`]: Now lint more expressions
  [#8958](https://github.com/rust-lang/rust-clippy/pull/8958)
* [`question_mark`]: Now works for simple `if let` expressions
  [#8356](https://github.com/rust-lang/rust-clippy/pull/8356)
* [`undocumented_unsafe_blocks`]: Now finds comments before the start of closures
  [#9117](https://github.com/rust-lang/rust-clippy/pull/9117)
* [`trait_duplication_in_bounds`]: Now catches duplicate bounds in where clauses
  [#8703](https://github.com/rust-lang/rust-clippy/pull/8703)
* [`shadow_reuse`], [`shadow_same`], [`shadow_unrelated`]: Now lint in const blocks
  [#9124](https://github.com/rust-lang/rust-clippy/pull/9124)
* [`slow_vector_initialization`]: Now detects cases with `vec.capacity()`
  [#8953](https://github.com/rust-lang/rust-clippy/pull/8953)
* [`unused_self`]: Now respects the `avoid-breaking-exported-api` config option
  [#9199](https://github.com/rust-lang/rust-clippy/pull/9199)
* [`box_collection`]: Now supports all std collections
  [#9170](https://github.com/rust-lang/rust-clippy/pull/9170)

### False Positive Fixes

* [`significant_drop_in_scrutinee`]: Now ignores calls to `IntoIterator::into_iter`
  [#9140](https://github.com/rust-lang/rust-clippy/pull/9140)
* [`while_let_loop`]: Now ignores cases when the significant drop order would change
  [#8981](https://github.com/rust-lang/rust-clippy/pull/8981)
* [`branches_sharing_code`]: Now ignores cases where moved variables have a significant
  drop or variable modifications can affect the conditions
  [#9138](https://github.com/rust-lang/rust-clippy/pull/9138)
* [`let_underscore_lock`]: Now ignores bindings that aren't locked
  [#8990](https://github.com/rust-lang/rust-clippy/pull/8990)
* [`trivially_copy_pass_by_ref`]: Now tracks lifetimes and ignores cases where unsafe
  pointers are used
  [#8639](https://github.com/rust-lang/rust-clippy/pull/8639)
* [`let_unit_value`]: No longer ignores `#[allow]` attributes on the value
  [#9082](https://github.com/rust-lang/rust-clippy/pull/9082)
* [`declare_interior_mutable_const`]: Now ignores the `thread_local!` macro
  [#9015](https://github.com/rust-lang/rust-clippy/pull/9015)
* [`if_same_then_else`]: Now ignores branches with `todo!` and `unimplemented!`
  [#9006](https://github.com/rust-lang/rust-clippy/pull/9006)
* [`enum_variant_names`]: Now ignores names with `_` prefixes
  [#9032](https://github.com/rust-lang/rust-clippy/pull/9032)
* [`let_unit_value`]: Now ignores cases, where the unit type is manually specified
  [#9056](https://github.com/rust-lang/rust-clippy/pull/9056)
* [`match_same_arms`]: Now ignores branches with `todo!`
  [#9207](https://github.com/rust-lang/rust-clippy/pull/9207)
* [`assign_op_pattern`]: Ignores cases that break borrowing rules
  [#9214](https://github.com/rust-lang/rust-clippy/pull/9214)
* [`extra_unused_lifetimes`]: No longer triggers in derive macros
  [#9037](https://github.com/rust-lang/rust-clippy/pull/9037)
* [`mismatching_type_param_order`]: Now ignores complicated generic parameters
  [#9146](https://github.com/rust-lang/rust-clippy/pull/9146)
* [`equatable_if_let`]: No longer lints in macros
  [#9074](https://github.com/rust-lang/rust-clippy/pull/9074)
* [`new_without_default`]: Now ignores generics and lifetime parameters on `fn new`
  [#9115](https://github.com/rust-lang/rust-clippy/pull/9115)
* [`needless_borrow`]: Now ignores cases that result in the execution of different traits
  [#9096](https://github.com/rust-lang/rust-clippy/pull/9096)
* [`declare_interior_mutable_const`]: No longer triggers in thread-local initializers
  [#9246](https://github.com/rust-lang/rust-clippy/pull/9246)

### Suggestion Fixes/Improvements

* [`type_repetition_in_bounds`]: The suggestion now works with maybe bounds
  [#9132](https://github.com/rust-lang/rust-clippy/pull/9132)
* [`transmute_ptr_to_ref`]: Now suggests `pointer::cast` when possible
  [#8939](https://github.com/rust-lang/rust-clippy/pull/8939)
* [`useless_format`]: Now suggests the correct variable name
  [#9237](https://github.com/rust-lang/rust-clippy/pull/9237)
* [`or_fun_call`]: The lint emission will now only span over the `unwrap_or` call
  [#9144](https://github.com/rust-lang/rust-clippy/pull/9144)
* [`neg_multiply`]: Now suggests adding parentheses around suggestion if needed
  [#9026](https://github.com/rust-lang/rust-clippy/pull/9026)
* [`unnecessary_lazy_evaluations`]: Now suggest for `bool::then_some` for lazy evaluation
  [#9099](https://github.com/rust-lang/rust-clippy/pull/9099)
* [`manual_flatten`]: Improved message for long code snippets
  [#9156](https://github.com/rust-lang/rust-clippy/pull/9156)
* [`explicit_counter_loop`]: The suggestion is now machine applicable
  [#9149](https://github.com/rust-lang/rust-clippy/pull/9149)
* [`needless_borrow`]: Now keeps parentheses around fields, when needed
  [#9210](https://github.com/rust-lang/rust-clippy/pull/9210)
* [`while_let_on_iterator`]: The suggestion now works in `FnOnce` closures
  [#9134](https://github.com/rust-lang/rust-clippy/pull/9134)

### ICE Fixes

* Fix ICEs related to `#![feature(generic_const_exprs)]` usage
  [#9241](https://github.com/rust-lang/rust-clippy/pull/9241)
* Fix ICEs related to reference lints
  [#9093](https://github.com/rust-lang/rust-clippy/pull/9093)
* [`question_mark`]: Fix ICE on zero field tuple structs
  [#9244](https://github.com/rust-lang/rust-clippy/pull/9244)

### Documentation Improvements

* [`needless_option_take`]: Now includes a "What it does" and "Why is this bad?" section.
  [#9022](https://github.com/rust-lang/rust-clippy/pull/9022)

### Others

* Using `--cap-lints=allow` and only `--force-warn`ing some will now work with Clippy's driver
  [#9036](https://github.com/rust-lang/rust-clippy/pull/9036)
* Clippy now tries to read the `rust-version` from `Cargo.toml` to identify the
  minimum supported rust version
  [#8774](https://github.com/rust-lang/rust-clippy/pull/8774)

## Rust 1.63

Released 2022-08-11

[View all 91 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2022-05-05T17%3A24%3A22Z..2022-06-16T14%3A24%3A48Z+base%3Amaster)

### New Lints

* [`borrow_deref_ref`]
  [#7930](https://github.com/rust-lang/rust-clippy/pull/7930)
* [`doc_link_with_quotes`]
  [#8385](https://github.com/rust-lang/rust-clippy/pull/8385)
* [`no_effect_replace`]
  [#8754](https://github.com/rust-lang/rust-clippy/pull/8754)
* [`rc_clone_in_vec_init`]
  [#8769](https://github.com/rust-lang/rust-clippy/pull/8769)
* [`derive_partial_eq_without_eq`]
  [#8796](https://github.com/rust-lang/rust-clippy/pull/8796)
* [`mismatching_type_param_order`]
  [#8831](https://github.com/rust-lang/rust-clippy/pull/8831)
* [`duplicate_mod`] [#8832](https://github.com/rust-lang/rust-clippy/pull/8832)
* [`unused_rounding`]
  [#8866](https://github.com/rust-lang/rust-clippy/pull/8866)
* [`get_first`] [#8882](https://github.com/rust-lang/rust-clippy/pull/8882)
* [`swap_ptr_to_ref`]
  [#8916](https://github.com/rust-lang/rust-clippy/pull/8916)
* [`almost_complete_letter_range`]
  [#8918](https://github.com/rust-lang/rust-clippy/pull/8918)
* [`needless_parens_on_range_literals`]
  [#8933](https://github.com/rust-lang/rust-clippy/pull/8933)
* [`as_underscore`] [#8934](https://github.com/rust-lang/rust-clippy/pull/8934)

### Moves and Deprecations

* Rename `eval_order_dependence` to [`mixed_read_write_in_expression`], move to
  `nursery` [#8621](https://github.com/rust-lang/rust-clippy/pull/8621)

### Enhancements

* [`undocumented_unsafe_blocks`]: Now also lints on unsafe trait implementations
  [#8761](https://github.com/rust-lang/rust-clippy/pull/8761)
* [`empty_line_after_outer_attr`]: Now also lints on argumentless macros
  [#8790](https://github.com/rust-lang/rust-clippy/pull/8790)
* [`expect_used`]: Now can be disabled in tests with the `allow-expect-in-tests`
  option [#8802](https://github.com/rust-lang/rust-clippy/pull/8802)
* [`unwrap_used`]: Now can be disabled in tests with the `allow-unwrap-in-tests`
  option [#8802](https://github.com/rust-lang/rust-clippy/pull/8802)
* [`disallowed_methods`]: Now also lints indirect usages
  [#8852](https://github.com/rust-lang/rust-clippy/pull/8852)
* [`get_last_with_len`]: Now also lints `VecDeque` and any deref to slice
  [#8862](https://github.com/rust-lang/rust-clippy/pull/8862)
* [`manual_range_contains`]: Now also lints on chains of `&&` and `||`
  [#8884](https://github.com/rust-lang/rust-clippy/pull/8884)
* [`rc_clone_in_vec_init`]: Now also lints on `Weak`
  [#8885](https://github.com/rust-lang/rust-clippy/pull/8885)
* [`dbg_macro`]: Introduce `allow-dbg-in-tests` config option
  [#8897](https://github.com/rust-lang/rust-clippy/pull/8897)
* [`use_self`]: Now also lints on `TupleStruct` and `Struct` patterns
  [#8899](https://github.com/rust-lang/rust-clippy/pull/8899)
* [`manual_find_map`] and [`manual_filter_map`]: Now also lints on more complex
  method chains inside `map`
  [#8930](https://github.com/rust-lang/rust-clippy/pull/8930)
* [`needless_return`]: Now also lints on macro expressions in return statements
  [#8932](https://github.com/rust-lang/rust-clippy/pull/8932)
* [`doc_markdown`]: Users can now indicate, that the `doc-valid-idents` config
  should extend the default and not replace it
  [#8944](https://github.com/rust-lang/rust-clippy/pull/8944)
* [`disallowed_names`]: Users can now indicate, that the `disallowed-names`
  config should extend the default and not replace it
  [#8944](https://github.com/rust-lang/rust-clippy/pull/8944)
* [`never_loop`]: Now checks for `continue` in struct expression
  [#9002](https://github.com/rust-lang/rust-clippy/pull/9002)

### False Positive Fixes

* [`useless_transmute`]: No longer lints on types with erased regions
  [#8564](https://github.com/rust-lang/rust-clippy/pull/8564)
* [`vec_init_then_push`]: No longer lints when further extended
  [#8699](https://github.com/rust-lang/rust-clippy/pull/8699)
* [`cmp_owned`]: No longer lints on `From::from` for `Copy` types
  [#8807](https://github.com/rust-lang/rust-clippy/pull/8807)
* [`redundant_allocation`]: No longer lints on fat pointers that would become
  thin pointers [#8813](https://github.com/rust-lang/rust-clippy/pull/8813)
* [`derive_partial_eq_without_eq`]:
    * Handle differing predicates applied by `#[derive(PartialEq)]` and
      `#[derive(Eq)]`
      [#8869](https://github.com/rust-lang/rust-clippy/pull/8869)
    * No longer lints on non-public types and better handles generics
      [#8950](https://github.com/rust-lang/rust-clippy/pull/8950)
* [`empty_line_after_outer_attr`]: No longer lints empty lines in inner
  string values [#8892](https://github.com/rust-lang/rust-clippy/pull/8892)
* [`branches_sharing_code`]: No longer lints when using different binding names
  [#8901](https://github.com/rust-lang/rust-clippy/pull/8901)
* [`significant_drop_in_scrutinee`]: No longer lints on Try `?` and `await`
  desugared expressions [#8902](https://github.com/rust-lang/rust-clippy/pull/8902)
* [`checked_conversions`]: No longer lints in `const` contexts
  [#8907](https://github.com/rust-lang/rust-clippy/pull/8907)
* [`iter_overeager_cloned`]: No longer lints on `.cloned().flatten()` when
  `T::Item` doesn't implement `IntoIterator`
  [#8960](https://github.com/rust-lang/rust-clippy/pull/8960)

### Suggestion Fixes/Improvements

* [`vec_init_then_push`]: Suggest to remove `mut` binding when possible
  [#8699](https://github.com/rust-lang/rust-clippy/pull/8699)
* [`manual_range_contains`]: Fix suggestion for integers with different signs
  [#8763](https://github.com/rust-lang/rust-clippy/pull/8763)
* [`identity_op`]: Add parenthesis to suggestions where required
  [#8786](https://github.com/rust-lang/rust-clippy/pull/8786)
* [`cast_lossless`]: No longer gives wrong suggestion on `usize`/`isize`->`f64`
  [#8778](https://github.com/rust-lang/rust-clippy/pull/8778)
* [`rc_clone_in_vec_init`]: Add suggestion
  [#8814](https://github.com/rust-lang/rust-clippy/pull/8814)
* The "unknown field" error messages for config files now wraps the field names
  [#8823](https://github.com/rust-lang/rust-clippy/pull/8823)
* [`cast_abs_to_unsigned`]: Do not remove cast if it's required
  [#8876](https://github.com/rust-lang/rust-clippy/pull/8876)
* [`significant_drop_in_scrutinee`]: Improve lint message for types that are not
  references and not trivially clone-able
  [#8902](https://github.com/rust-lang/rust-clippy/pull/8902)
* [`for_loops_over_fallibles`]: Now suggests the correct variant of `iter()`,
  `iter_mut()` or `into_iter()`
  [#8941](https://github.com/rust-lang/rust-clippy/pull/8941)

### ICE Fixes

* Fix ICE in [`let_unit_value`] when calling a `static`/`const` callable type
  [#8835](https://github.com/rust-lang/rust-clippy/pull/8835)
* Fix ICEs on callable `static`/`const`s
  [#8896](https://github.com/rust-lang/rust-clippy/pull/8896)
* [`needless_late_init`]
  [#8912](https://github.com/rust-lang/rust-clippy/pull/8912)
* Fix ICE in shadow lints
  [#8913](https://github.com/rust-lang/rust-clippy/pull/8913)

### Documentation Improvements

* Clippy has a [Book](https://doc.rust-lang.org/nightly/clippy/) now!
  [#7359](https://github.com/rust-lang/rust-clippy/pull/7359)
* Add a *copy lint name*-button to Clippy's lint list
  [#8839](https://github.com/rust-lang/rust-clippy/pull/8839)
* Display past names of renamed lints on Clippy's lint list
  [#8843](https://github.com/rust-lang/rust-clippy/pull/8843)
* Add the ability to show the lint output in the lint list
  [#8947](https://github.com/rust-lang/rust-clippy/pull/8947)

## Rust 1.62

Released 2022-06-30

[View all 90 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2022-03-25T17%3A22%3A30Z..2022-05-05T13%3A29%3A44Z+base%3Amaster)

### New Lints

* [`large_include_file`]
  [#8727](https://github.com/rust-lang/rust-clippy/pull/8727)
* [`cast_abs_to_unsigned`]
  [#8635](https://github.com/rust-lang/rust-clippy/pull/8635)
* [`err_expect`]
  [#8606](https://github.com/rust-lang/rust-clippy/pull/8606)
* [`unnecessary_owned_empty_strings`]
  [#8660](https://github.com/rust-lang/rust-clippy/pull/8660)
* [`empty_structs_with_brackets`]
  [#8594](https://github.com/rust-lang/rust-clippy/pull/8594)
* [`crate_in_macro_def`]
  [#8576](https://github.com/rust-lang/rust-clippy/pull/8576)
* [`needless_option_take`]
  [#8665](https://github.com/rust-lang/rust-clippy/pull/8665)
* [`bytes_count_to_len`]
  [#8711](https://github.com/rust-lang/rust-clippy/pull/8711)
* [`is_digit_ascii_radix`]
  [#8624](https://github.com/rust-lang/rust-clippy/pull/8624)
* [`await_holding_invalid_type`]
  [#8707](https://github.com/rust-lang/rust-clippy/pull/8707)
* [`trim_split_whitespace`]
  [#8575](https://github.com/rust-lang/rust-clippy/pull/8575)
* [`pub_use`]
  [#8670](https://github.com/rust-lang/rust-clippy/pull/8670)
* [`format_push_string`]
  [#8626](https://github.com/rust-lang/rust-clippy/pull/8626)
* [`empty_drop`]
  [#8571](https://github.com/rust-lang/rust-clippy/pull/8571)
* [`drop_non_drop`]
  [#8630](https://github.com/rust-lang/rust-clippy/pull/8630)
* [`forget_non_drop`]
  [#8630](https://github.com/rust-lang/rust-clippy/pull/8630)

### Moves and Deprecations

* Move [`only_used_in_recursion`] to `nursery` (now allow-by-default)
  [#8783](https://github.com/rust-lang/rust-clippy/pull/8783)
* Move [`stable_sort_primitive`] to `pedantic` (now allow-by-default)
  [#8716](https://github.com/rust-lang/rust-clippy/pull/8716)

### Enhancements

* Remove overlap between [`manual_split_once`] and [`needless_splitn`]
  [#8631](https://github.com/rust-lang/rust-clippy/pull/8631)
* [`map_identity`]: Now checks for needless `map_err`
  [#8487](https://github.com/rust-lang/rust-clippy/pull/8487)
* [`extra_unused_lifetimes`]: Now checks for impl lifetimes
  [#8737](https://github.com/rust-lang/rust-clippy/pull/8737)
* [`cast_possible_truncation`]: Now catches more cases with larger shift or divide operations
  [#8687](https://github.com/rust-lang/rust-clippy/pull/8687)
* [`identity_op`]: Now checks for modulo expressions
  [#8519](https://github.com/rust-lang/rust-clippy/pull/8519)
* [`panic`]: No longer lint in constant context
  [#8592](https://github.com/rust-lang/rust-clippy/pull/8592)
* [`manual_split_once`]: Now lints manual iteration of `splitn`
  [#8717](https://github.com/rust-lang/rust-clippy/pull/8717)
* [`self_named_module_files`], [`mod_module_files`]: Now handle relative module paths
  [#8611](https://github.com/rust-lang/rust-clippy/pull/8611)
* [`unsound_collection_transmute`]: Now has better size and alignment checks
  [#8648](https://github.com/rust-lang/rust-clippy/pull/8648)
* [`unnested_or_patterns`]: Ignore cases, where the suggestion would be longer
  [#8619](https://github.com/rust-lang/rust-clippy/pull/8619)

### False Positive Fixes

* [`rest_pat_in_fully_bound_structs`]: Now ignores structs marked with `#[non_exhaustive]`
  [#8690](https://github.com/rust-lang/rust-clippy/pull/8690)
* [`needless_late_init`]: No longer lints `if let` statements, `let mut` bindings or instances that
  changes the drop order significantly
  [#8617](https://github.com/rust-lang/rust-clippy/pull/8617)
* [`unnecessary_cast`]: No longer lints to casts to aliased or non-primitive types
  [#8596](https://github.com/rust-lang/rust-clippy/pull/8596)
* [`init_numbered_fields`]: No longer lints type aliases
  [#8780](https://github.com/rust-lang/rust-clippy/pull/8780)
* [`needless_option_as_deref`]: No longer lints for `as_deref_mut` on `Option` values that can't be moved
  [#8646](https://github.com/rust-lang/rust-clippy/pull/8646)
* [`mistyped_literal_suffixes`]: Now ignores float literals without an exponent
  [#8742](https://github.com/rust-lang/rust-clippy/pull/8742)
* [`undocumented_unsafe_blocks`]: Now ignores unsafe blocks from proc-macros and works better for sub-expressions
  [#8450](https://github.com/rust-lang/rust-clippy/pull/8450)
* [`same_functions_in_if_condition`]: Now allows different constants, even if they have the same value
  [#8673](https://github.com/rust-lang/rust-clippy/pull/8673)
* [`needless_match`]: Now checks for more complex types and ignores type coercion
  [#8549](https://github.com/rust-lang/rust-clippy/pull/8549)
* [`assertions_on_constants`]: Now ignores constants from `cfg!` macros
  [#8614](https://github.com/rust-lang/rust-clippy/pull/8614)
* [`indexing_slicing`]: Fix false positives with constant indices in
  [#8588](https://github.com/rust-lang/rust-clippy/pull/8588)
* [`iter_with_drain`]: Now ignores iterator references
  [#8668](https://github.com/rust-lang/rust-clippy/pull/8668)
* [`useless_attribute`]: Now allows [`redundant_pub_crate`] on `use` items
  [#8743](https://github.com/rust-lang/rust-clippy/pull/8743)
* [`cast_ptr_alignment`]: Now ignores expressions, when used for unaligned reads and writes
  [#8632](https://github.com/rust-lang/rust-clippy/pull/8632)
* [`wrong_self_convention`]: Now allows `&mut self` and no self as arguments for `is_*` methods
  [#8738](https://github.com/rust-lang/rust-clippy/pull/8738)
* [`mut_from_ref`]: Only lint in unsafe code
  [#8647](https://github.com/rust-lang/rust-clippy/pull/8647)
* [`redundant_pub_crate`]: Now allows macro exports
  [#8736](https://github.com/rust-lang/rust-clippy/pull/8736)
* [`needless_match`]: Ignores cases where the else block expression is different
  [#8700](https://github.com/rust-lang/rust-clippy/pull/8700)
* [`transmute_int_to_char`]: Now allows transmutations in `const` code
  [#8610](https://github.com/rust-lang/rust-clippy/pull/8610)
* [`manual_non_exhaustive`]: Ignores cases, where the enum value is used
  [#8645](https://github.com/rust-lang/rust-clippy/pull/8645)
* [`redundant_closure`]: Now ignores coerced closure
  [#8431](https://github.com/rust-lang/rust-clippy/pull/8431)
* [`identity_op`]: Is now ignored in cases where extra brackets would be needed
  [#8730](https://github.com/rust-lang/rust-clippy/pull/8730)
* [`let_unit_value`]: Now ignores cases which are used for type inference
  [#8563](https://github.com/rust-lang/rust-clippy/pull/8563)

### Suggestion Fixes/Improvements

* [`manual_split_once`]: Fixed incorrect suggestions for single result accesses
  [#8631](https://github.com/rust-lang/rust-clippy/pull/8631)
* [`bytes_nth`]: Fix typos in the diagnostic message
  [#8403](https://github.com/rust-lang/rust-clippy/pull/8403)
* [`mistyped_literal_suffixes`]: Now suggests the correct integer types
  [#8742](https://github.com/rust-lang/rust-clippy/pull/8742)
* [`unnecessary_to_owned`]: Fixed suggestion based on the configured msrv
  [#8692](https://github.com/rust-lang/rust-clippy/pull/8692)
* [`single_element_loop`]: Improve lint for Edition 2021 arrays
  [#8616](https://github.com/rust-lang/rust-clippy/pull/8616)
* [`manual_bits`]: Now includes a cast for proper type conversion, when needed
  [#8677](https://github.com/rust-lang/rust-clippy/pull/8677)
* [`option_map_unit_fn`], [`result_map_unit_fn`]: Fix some incorrect suggestions
  [#8584](https://github.com/rust-lang/rust-clippy/pull/8584)
* [`collapsible_else_if`]: Add whitespace in suggestion
  [#8729](https://github.com/rust-lang/rust-clippy/pull/8729)
* [`transmute_bytes_to_str`]: Now suggest `from_utf8_unchecked` in `const` context
  [#8612](https://github.com/rust-lang/rust-clippy/pull/8612)
* [`map_clone`]: Improve message and suggestion based on the msrv
  [#8688](https://github.com/rust-lang/rust-clippy/pull/8688)
* [`needless_late_init`]: Now shows the `let` statement where it was first initialized
  [#8779](https://github.com/rust-lang/rust-clippy/pull/8779)

### ICE Fixes

* [`only_used_in_recursion`]
  [#8691](https://github.com/rust-lang/rust-clippy/pull/8691)
* [`cast_slice_different_sizes`]
  [#8720](https://github.com/rust-lang/rust-clippy/pull/8720)
* [`iter_overeager_cloned`]
  [#8602](https://github.com/rust-lang/rust-clippy/pull/8602)
* [`undocumented_unsafe_blocks`]
  [#8686](https://github.com/rust-lang/rust-clippy/pull/8686)

## Rust 1.61

Released 2022-05-19

[View all 60 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2022-02-11T16%3A54%3A41Z..2022-03-24T13%3A42%3A25Z+base%3Amaster)

### New Lints

* [`only_used_in_recursion`]
  [#8422](https://github.com/rust-lang/rust-clippy/pull/8422)
* [`cast_enum_truncation`]
  [#8381](https://github.com/rust-lang/rust-clippy/pull/8381)
* [`missing_spin_loop`]
  [#8174](https://github.com/rust-lang/rust-clippy/pull/8174)
* [`deref_by_slicing`]
  [#8218](https://github.com/rust-lang/rust-clippy/pull/8218)
* [`needless_match`]
  [#8471](https://github.com/rust-lang/rust-clippy/pull/8471)
* [`allow_attributes_without_reason`]
  [#8504](https://github.com/rust-lang/rust-clippy/pull/8504)
* [`print_in_format_impl`]
  [#8253](https://github.com/rust-lang/rust-clippy/pull/8253)
* [`unnecessary_find_map`]
  [#8489](https://github.com/rust-lang/rust-clippy/pull/8489)
* [`or_then_unwrap`]
  [#8561](https://github.com/rust-lang/rust-clippy/pull/8561)
* [`unnecessary_join`]
  [#8579](https://github.com/rust-lang/rust-clippy/pull/8579)
* [`iter_with_drain`]
  [#8483](https://github.com/rust-lang/rust-clippy/pull/8483)
* [`cast_enum_constructor`]
  [#8562](https://github.com/rust-lang/rust-clippy/pull/8562)
* [`cast_slice_different_sizes`]
  [#8445](https://github.com/rust-lang/rust-clippy/pull/8445)

### Moves and Deprecations

* Moved [`transmute_undefined_repr`] to `nursery` (now allow-by-default)
  [#8432](https://github.com/rust-lang/rust-clippy/pull/8432)
* Moved [`try_err`] to `restriction`
  [#8544](https://github.com/rust-lang/rust-clippy/pull/8544)
* Move [`iter_with_drain`] to `nursery`
  [#8541](https://github.com/rust-lang/rust-clippy/pull/8541)
* Renamed `to_string_in_display` to [`recursive_format_impl`]
  [#8188](https://github.com/rust-lang/rust-clippy/pull/8188)

### Enhancements

* [`dbg_macro`]: The lint level can now be set with crate attributes and works inside macros
  [#8411](https://github.com/rust-lang/rust-clippy/pull/8411)
* [`ptr_as_ptr`]: Now works inside macros
  [#8442](https://github.com/rust-lang/rust-clippy/pull/8442)
* [`use_self`]: Now works for variants in match expressions
  [#8456](https://github.com/rust-lang/rust-clippy/pull/8456)
* [`await_holding_lock`]: Now lints for `parking_lot::{Mutex, RwLock}`
  [#8419](https://github.com/rust-lang/rust-clippy/pull/8419)
* [`recursive_format_impl`]: Now checks for format calls on `self`
  [#8188](https://github.com/rust-lang/rust-clippy/pull/8188)

### False Positive Fixes

* [`new_without_default`]: No longer lints for `new()` methods with `#[doc(hidden)]`
  [#8472](https://github.com/rust-lang/rust-clippy/pull/8472)
* [`transmute_undefined_repr`]: No longer lints for single field structs with `#[repr(C)]`,
  generic parameters, wide pointers, unions, tuples and allow several forms of type erasure
  [#8425](https://github.com/rust-lang/rust-clippy/pull/8425)
  [#8553](https://github.com/rust-lang/rust-clippy/pull/8553)
  [#8440](https://github.com/rust-lang/rust-clippy/pull/8440)
  [#8547](https://github.com/rust-lang/rust-clippy/pull/8547)
* [`match_single_binding`], [`match_same_arms`], [`match_as_ref`], [`match_bool`]: No longer
  lint `match` expressions with `cfg`ed arms
  [#8443](https://github.com/rust-lang/rust-clippy/pull/8443)
* [`single_component_path_imports`]: No longer lint on macros
  [#8537](https://github.com/rust-lang/rust-clippy/pull/8537)
* [`ptr_arg`]: Allow `&mut` arguments for `Cow<_>`
  [#8552](https://github.com/rust-lang/rust-clippy/pull/8552)
* [`needless_borrow`]: No longer lints for method calls
  [#8441](https://github.com/rust-lang/rust-clippy/pull/8441)
* [`match_same_arms`]: Now ensures that interposing arm patterns don't overlap
  [#8232](https://github.com/rust-lang/rust-clippy/pull/8232)
* [`default_trait_access`]: Now allows `Default::default` in update expressions
  [#8433](https://github.com/rust-lang/rust-clippy/pull/8433)

### Suggestion Fixes/Improvements

* [`redundant_slicing`]: Fixed suggestion for a method calls
  [#8218](https://github.com/rust-lang/rust-clippy/pull/8218)
* [`map_flatten`]: Long suggestions will now be split up into two help messages
  [#8520](https://github.com/rust-lang/rust-clippy/pull/8520)
* [`unnecessary_lazy_evaluations`]: Now shows suggestions for longer code snippets
  [#8543](https://github.com/rust-lang/rust-clippy/pull/8543)
* [`unnecessary_sort_by`]: Now suggests `Reverse` including the path
  [#8462](https://github.com/rust-lang/rust-clippy/pull/8462)
* [`search_is_some`]: More suggestions are now `MachineApplicable`
  [#8536](https://github.com/rust-lang/rust-clippy/pull/8536)

### Documentation Improvements

* [`new_without_default`]: Document `pub` requirement for the struct and fields
  [#8429](https://github.com/rust-lang/rust-clippy/pull/8429)

## Rust 1.60

Released 2022-04-07

[View all 73 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2021-12-31T17%3A53%3A37Z..2022-02-10T17%3A31%3A37Z+base%3Amaster)

### New Lints

* [`single_char_lifetime_names`]
  [#8236](https://github.com/rust-lang/rust-clippy/pull/8236)
* [`iter_overeager_cloned`]
  [#8203](https://github.com/rust-lang/rust-clippy/pull/8203)
* [`transmute_undefined_repr`]
  [#8398](https://github.com/rust-lang/rust-clippy/pull/8398)
* [`default_union_representation`]
  [#8289](https://github.com/rust-lang/rust-clippy/pull/8289)
* [`manual_bits`]
  [#8213](https://github.com/rust-lang/rust-clippy/pull/8213)
* [`borrow_as_ptr`]
  [#8210](https://github.com/rust-lang/rust-clippy/pull/8210)

### Moves and Deprecations

* Moved [`disallowed_methods`] and [`disallowed_types`] to `style` (now warn-by-default)
  [#8261](https://github.com/rust-lang/rust-clippy/pull/8261)
* Rename `ref_in_deref` to [`needless_borrow`]
  [#8217](https://github.com/rust-lang/rust-clippy/pull/8217)
* Moved [`mutex_atomic`] to `nursery` (now allow-by-default)
  [#8260](https://github.com/rust-lang/rust-clippy/pull/8260)

### Enhancements

* [`ptr_arg`]: Now takes the argument usage into account and lints for mutable references
  [#8271](https://github.com/rust-lang/rust-clippy/pull/8271)
* [`unused_io_amount`]: Now supports async read and write traits
  [#8179](https://github.com/rust-lang/rust-clippy/pull/8179)
* [`while_let_on_iterator`]: Improved detection to catch more cases
  [#8221](https://github.com/rust-lang/rust-clippy/pull/8221)
* [`trait_duplication_in_bounds`]: Now covers trait functions with `Self` bounds
  [#8252](https://github.com/rust-lang/rust-clippy/pull/8252)
* [`unwrap_used`]: Now works for `.get(i).unwrap()` and `.get_mut(i).unwrap()`
  [#8372](https://github.com/rust-lang/rust-clippy/pull/8372)
* [`map_clone`]: The suggestion takes `msrv` into account
  [#8280](https://github.com/rust-lang/rust-clippy/pull/8280)
* [`manual_bits`] and [`borrow_as_ptr`]: Now track the `clippy::msrv` attribute
  [#8280](https://github.com/rust-lang/rust-clippy/pull/8280)
* [`disallowed_methods`]: Now works for methods on primitive types
  [#8112](https://github.com/rust-lang/rust-clippy/pull/8112)
* [`not_unsafe_ptr_arg_deref`]: Now works for type aliases
  [#8273](https://github.com/rust-lang/rust-clippy/pull/8273)
* [`needless_question_mark`]: Now works for async functions
  [#8311](https://github.com/rust-lang/rust-clippy/pull/8311)
* [`iter_not_returning_iterator`]: Now handles type projections
  [#8228](https://github.com/rust-lang/rust-clippy/pull/8228)
* [`wrong_self_convention`]: Now detects wrong `self` references in more cases
  [#8208](https://github.com/rust-lang/rust-clippy/pull/8208)
* [`single_match`]: Now works for `match` statements with tuples
  [#8322](https://github.com/rust-lang/rust-clippy/pull/8322)

### False Positive Fixes

* [`erasing_op`]: No longer triggers if the output type changes
  [#8204](https://github.com/rust-lang/rust-clippy/pull/8204)
* [`if_same_then_else`]: No longer triggers for `if let` statements
  [#8297](https://github.com/rust-lang/rust-clippy/pull/8297)
* [`manual_memcpy`]: No longer lints on `VecDeque`
  [#8226](https://github.com/rust-lang/rust-clippy/pull/8226)
* [`trait_duplication_in_bounds`]: Now takes path segments into account
  [#8315](https://github.com/rust-lang/rust-clippy/pull/8315)
* [`deref_addrof`]: No longer lints when the dereference or borrow occurs in different a context
  [#8268](https://github.com/rust-lang/rust-clippy/pull/8268)
* [`type_repetition_in_bounds`]: Now checks for full equality to prevent false positives
  [#8224](https://github.com/rust-lang/rust-clippy/pull/8224)
* [`ptr_arg`]: No longer lint for mutable references in traits
  [#8369](https://github.com/rust-lang/rust-clippy/pull/8369)
* [`implicit_clone`]: No longer lints for double references
  [#8231](https://github.com/rust-lang/rust-clippy/pull/8231)
* [`needless_lifetimes`]: No longer lints lifetimes for explicit `self` types
  [#8278](https://github.com/rust-lang/rust-clippy/pull/8278)
* [`op_ref`]: No longer lints in `BinOp` impl if that can cause recursion
  [#8298](https://github.com/rust-lang/rust-clippy/pull/8298)
* [`enum_variant_names`]: No longer triggers for empty variant names
  [#8329](https://github.com/rust-lang/rust-clippy/pull/8329)
* [`redundant_closure`]: No longer lints for `Arc<T>` or `Rc<T>`
  [#8193](https://github.com/rust-lang/rust-clippy/pull/8193)
* [`iter_not_returning_iterator`]: No longer lints on trait implementations but therefore on trait definitions
  [#8228](https://github.com/rust-lang/rust-clippy/pull/8228)
* [`single_match`]: No longer lints on exhaustive enum patterns without a wildcard
  [#8322](https://github.com/rust-lang/rust-clippy/pull/8322)
* [`manual_swap`]: No longer lints on cases that involve automatic dereferences
  [#8220](https://github.com/rust-lang/rust-clippy/pull/8220)
* [`useless_format`]: Now works for implicit named arguments
  [#8295](https://github.com/rust-lang/rust-clippy/pull/8295)

### Suggestion Fixes/Improvements

* [`needless_borrow`]: Prevent mutable borrows being moved and suggest removing the borrow on method calls
  [#8217](https://github.com/rust-lang/rust-clippy/pull/8217)
* [`chars_next_cmp`]: Correctly escapes the suggestion
  [#8376](https://github.com/rust-lang/rust-clippy/pull/8376)
* [`explicit_write`]: Add suggestions for `write!`s with format arguments
  [#8365](https://github.com/rust-lang/rust-clippy/pull/8365)
* [`manual_memcpy`]: Suggests `copy_from_slice` when applicable
  [#8226](https://github.com/rust-lang/rust-clippy/pull/8226)
* [`or_fun_call`]: Improved suggestion display for long arguments
  [#8292](https://github.com/rust-lang/rust-clippy/pull/8292)
* [`unnecessary_cast`]: Now correctly includes the sign
  [#8350](https://github.com/rust-lang/rust-clippy/pull/8350)
* [`cmp_owned`]: No longer flips the comparison order
  [#8299](https://github.com/rust-lang/rust-clippy/pull/8299)
* [`explicit_counter_loop`]: Now correctly suggests `iter()` on references
  [#8382](https://github.com/rust-lang/rust-clippy/pull/8382)

### ICE Fixes

* [`manual_split_once`]
  [#8250](https://github.com/rust-lang/rust-clippy/pull/8250)

### Documentation Improvements

* [`map_flatten`]: Add documentation for the `Option` type
  [#8354](https://github.com/rust-lang/rust-clippy/pull/8354)
* Document that Clippy's driver might use a different code generation than rustc
  [#8037](https://github.com/rust-lang/rust-clippy/pull/8037)
* Clippy's lint list will now automatically focus the search box
  [#8343](https://github.com/rust-lang/rust-clippy/pull/8343)

### Others

* Clippy now warns if we find multiple Clippy config files exist
  [#8326](https://github.com/rust-lang/rust-clippy/pull/8326)

## Rust 1.59

Released 2022-02-24

[View all 94 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2021-11-04T12%3A40%3A18Z..2021-12-30T13%3A36%3A20Z+base%3Amaster)

### New Lints

* [`index_refutable_slice`]
  [#7643](https://github.com/rust-lang/rust-clippy/pull/7643)
* [`needless_splitn`]
  [#7896](https://github.com/rust-lang/rust-clippy/pull/7896)
* [`unnecessary_to_owned`]
  [#7978](https://github.com/rust-lang/rust-clippy/pull/7978)
* [`needless_late_init`]
  [#7995](https://github.com/rust-lang/rust-clippy/pull/7995)
* [`octal_escapes`] [#8007](https://github.com/rust-lang/rust-clippy/pull/8007)
* [`return_self_not_must_use`]
  [#8071](https://github.com/rust-lang/rust-clippy/pull/8071)
* [`init_numbered_fields`]
  [#8170](https://github.com/rust-lang/rust-clippy/pull/8170)

### Moves and Deprecations

* Move `if_then_panic` to `pedantic` and rename to [`manual_assert`] (now
  allow-by-default) [#7810](https://github.com/rust-lang/rust-clippy/pull/7810)
* Rename `disallow_type` to [`disallowed_types`] and `disallowed_method` to
  [`disallowed_methods`]
  [#7984](https://github.com/rust-lang/rust-clippy/pull/7984)
* Move [`map_flatten`] to `complexity` (now warn-by-default)
  [#8054](https://github.com/rust-lang/rust-clippy/pull/8054)

### Enhancements

* [`match_overlapping_arm`]: Fix false negative where after included ranges,
  overlapping ranges weren't linted anymore
  [#7909](https://github.com/rust-lang/rust-clippy/pull/7909)
* [`deprecated_cfg_attr`]: Now takes the specified MSRV into account
  [#7944](https://github.com/rust-lang/rust-clippy/pull/7944)
* [`cast_lossless`]: Now also lints for `bool` to integer casts
  [#7948](https://github.com/rust-lang/rust-clippy/pull/7948)
* [`let_underscore_lock`]: Also emit lints for the `parking_lot` crate
  [#7957](https://github.com/rust-lang/rust-clippy/pull/7957)
* [`needless_borrow`]
  [#7977](https://github.com/rust-lang/rust-clippy/pull/7977)
    * Lint when a borrow is auto-dereffed more than once
    * Lint in the trailing expression of a block for a match arm
* [`strlen_on_c_strings`]
  [8001](https://github.com/rust-lang/rust-clippy/pull/8001)
    * Lint when used without a fully-qualified path
    * Suggest removing the surrounding unsafe block when possible
* [`non_ascii_literal`]: Now also lints on `char`s, not just `string`s
  [#8034](https://github.com/rust-lang/rust-clippy/pull/8034)
* [`single_char_pattern`]: Now also lints on `split_inclusive`, `split_once`,
  `rsplit_once`, `replace`, and `replacen`
  [#8077](https://github.com/rust-lang/rust-clippy/pull/8077)
* [`unwrap_or_else_default`]: Now also lints on `std` constructors like
  `Vec::new`, `HashSet::new`, and `HashMap::new`
  [#8163](https://github.com/rust-lang/rust-clippy/pull/8163)
* [`shadow_reuse`]: Now also lints on shadowed `if let` bindings, instead of
  [`shadow_unrelated`]
  [#8165](https://github.com/rust-lang/rust-clippy/pull/8165)

### False Positive Fixes

* [`or_fun_call`], [`unnecessary_lazy_evaluations`]: Improve heuristics, so that
  cheap functions (e.g. calling `.len()` on a `Vec`) won't get linted anymore
  [#7639](https://github.com/rust-lang/rust-clippy/pull/7639)
* [`manual_split_once`]: No longer suggests code changing the original behavior
  [#7896](https://github.com/rust-lang/rust-clippy/pull/7896)
* Don't show [`no_effect`] or [`unnecessary_operation`] warning for unit struct
  implementing `FnOnce`
  [#7898](https://github.com/rust-lang/rust-clippy/pull/7898)
* [`semicolon_if_nothing_returned`]: Fixed a bug, where the lint wrongly
  triggered on `let-else` statements
  [#7955](https://github.com/rust-lang/rust-clippy/pull/7955)
* [`if_then_some_else_none`]: No longer lints if there is an early return
  [#7980](https://github.com/rust-lang/rust-clippy/pull/7980)
* [`needless_collect`]: No longer suggests removal of `collect` when removal
  would create code requiring mutably borrowing a value multiple times
  [#7982](https://github.com/rust-lang/rust-clippy/pull/7982)
* [`shadow_same`]: Fix false positive for `async` function's params
  [#7997](https://github.com/rust-lang/rust-clippy/pull/7997)
* [`suboptimal_flops`]: No longer triggers in constant functions
  [#8009](https://github.com/rust-lang/rust-clippy/pull/8009)
* [`type_complexity`]: No longer lints on associated types in traits
  [#8030](https://github.com/rust-lang/rust-clippy/pull/8030)
* [`question_mark`]: No longer lints if returned object is not local
  [#8080](https://github.com/rust-lang/rust-clippy/pull/8080)
* [`option_if_let_else`]: No longer lint on complex sub-patterns
  [#8086](https://github.com/rust-lang/rust-clippy/pull/8086)
* [`blocks_in_if_conditions`]: No longer lints on empty closures
  [#8100](https://github.com/rust-lang/rust-clippy/pull/8100)
* [`enum_variant_names`]: No longer lint when first prefix is only a substring
  of a camel-case word
  [#8127](https://github.com/rust-lang/rust-clippy/pull/8127)
* [`identity_op`]: Only lint on integral operands
  [#8183](https://github.com/rust-lang/rust-clippy/pull/8183)

### Suggestion Fixes/Improvements

* [`search_is_some`]: Fix suggestion for `any()` not taking item by reference
  [#7463](https://github.com/rust-lang/rust-clippy/pull/7463)
* [`almost_swapped`]: Now detects if there is a `no_std` or `no_core` attribute
  and adapts the suggestion accordingly
  [#7877](https://github.com/rust-lang/rust-clippy/pull/7877)
* [`redundant_pattern_matching`]: Fix suggestion for deref expressions
  [#7949](https://github.com/rust-lang/rust-clippy/pull/7949)
* [`explicit_counter_loop`]: Now also produces a suggestion for non-`usize`
  types [#7950](https://github.com/rust-lang/rust-clippy/pull/7950)
* [`manual_map`]: Fix suggestion when used with unsafe functions and blocks
  [#7968](https://github.com/rust-lang/rust-clippy/pull/7968)
* [`option_map_or_none`]: Suggest `map` over `and_then` when possible
  [#7971](https://github.com/rust-lang/rust-clippy/pull/7971)
* [`option_if_let_else`]: No longer expands macros in the suggestion
  [#7974](https://github.com/rust-lang/rust-clippy/pull/7974)
* [`iter_cloned_collect`]: Suggest `copied` over `cloned` when possible
  [#8006](https://github.com/rust-lang/rust-clippy/pull/8006)
* [`doc_markdown`]: No longer uses inline hints to improve readability of
  suggestion [#8011](https://github.com/rust-lang/rust-clippy/pull/8011)
* [`needless_question_mark`]: Now better explains the suggestion
  [#8028](https://github.com/rust-lang/rust-clippy/pull/8028)
* [`single_char_pattern`]: Escape backslash `\` in suggestion
  [#8067](https://github.com/rust-lang/rust-clippy/pull/8067)
* [`needless_bool`]: Suggest `a != b` over `!(a == b)`
  [#8117](https://github.com/rust-lang/rust-clippy/pull/8117)
* [`iter_skip_next`]: Suggest to add a `mut` if it is necessary in order to
  apply this lints suggestion
  [#8133](https://github.com/rust-lang/rust-clippy/pull/8133)
* [`neg_multiply`]: Now produces a suggestion
  [#8144](https://github.com/rust-lang/rust-clippy/pull/8144)
* [`needless_return`]: Now suggests the unit type `()` over an empty block `{}`
  in match arms [#8185](https://github.com/rust-lang/rust-clippy/pull/8185)
* [`suboptimal_flops`]: Now gives a syntactically correct suggestion for
  `to_radians` and `to_degrees`
  [#8187](https://github.com/rust-lang/rust-clippy/pull/8187)

### ICE Fixes

* [`undocumented_unsafe_blocks`]
  [#7945](https://github.com/rust-lang/rust-clippy/pull/7945)
  [#7988](https://github.com/rust-lang/rust-clippy/pull/7988)
* [`unnecessary_cast`]
  [#8167](https://github.com/rust-lang/rust-clippy/pull/8167)

### Documentation Improvements

* [`print_stdout`], [`print_stderr`], [`dbg_macro`]: Document how the lint level
  can be changed crate-wide
  [#8040](https://github.com/rust-lang/rust-clippy/pull/8040)
* Added a note to the `README` that config changes don't apply to already
  compiled code [#8175](https://github.com/rust-lang/rust-clippy/pull/8175)

### Others

* [Clippy's lint
  list](https://rust-lang.github.io/rust-clippy/master/index.html) now displays
  the version a lint was added. :tada:
  [#7813](https://github.com/rust-lang/rust-clippy/pull/7813)
* New and improved issue templates
  [#8032](https://github.com/rust-lang/rust-clippy/pull/8032)
* _Dev:_ Add `cargo dev lint` command, to run your modified Clippy version on a
  file [#7917](https://github.com/rust-lang/rust-clippy/pull/7917)

## Rust 1.58

Released 2022-01-13

[View all 68 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2021-10-07T09%3A49%3A18Z..2021-11-04T12%3A20%3A12Z+base%3Amaster)

### Rust 1.58.1

* Move [`non_send_fields_in_send_ty`] to `nursery` (now allow-by-default)
  [#8075](https://github.com/rust-lang/rust-clippy/pull/8075)
* [`useless_format`]: Handle implicit named arguments
  [#8295](https://github.com/rust-lang/rust-clippy/pull/8295)

### New lints

* [`transmute_num_to_bytes`]
  [#7805](https://github.com/rust-lang/rust-clippy/pull/7805)
* [`match_str_case_mismatch`]
  [#7806](https://github.com/rust-lang/rust-clippy/pull/7806)
* [`format_in_format_args`], [`to_string_in_format_args`]
  [#7743](https://github.com/rust-lang/rust-clippy/pull/7743)
* [`uninit_vec`]
  [#7682](https://github.com/rust-lang/rust-clippy/pull/7682)
* [`fn_to_numeric_cast_any`]
  [#7705](https://github.com/rust-lang/rust-clippy/pull/7705)
* [`undocumented_unsafe_blocks`]
  [#7748](https://github.com/rust-lang/rust-clippy/pull/7748)
* [`trailing_empty_array`]
  [#7838](https://github.com/rust-lang/rust-clippy/pull/7838)
* [`string_slice`]
  [#7878](https://github.com/rust-lang/rust-clippy/pull/7878)

### Moves or deprecations of lints

* Move [`non_send_fields_in_send_ty`] to `suspicious`
  [#7874](https://github.com/rust-lang/rust-clippy/pull/7874)
* Move [`non_ascii_literal`] to `restriction`
  [#7907](https://github.com/rust-lang/rust-clippy/pull/7907)

### Changes that expand what code existing lints cover

* [`question_mark`] now covers `Result`
  [#7840](https://github.com/rust-lang/rust-clippy/pull/7840)
* Make [`useless_format`] recognize bare `format!("")`
  [#7801](https://github.com/rust-lang/rust-clippy/pull/7801)
* Lint on underscored variables with no side effects in [`no_effect`]
  [#7775](https://github.com/rust-lang/rust-clippy/pull/7775)
* Expand [`match_ref_pats`] to check for multiple reference patterns
  [#7800](https://github.com/rust-lang/rust-clippy/pull/7800)

### False positive fixes

* Fix false positive of [`implicit_saturating_sub`] with `else` clause
  [#7832](https://github.com/rust-lang/rust-clippy/pull/7832)
* Fix [`question_mark`] when there is call in conditional predicate
  [#7860](https://github.com/rust-lang/rust-clippy/pull/7860)
* [`mut_mut`] no longer lints when type is defined in external macros
  [#7795](https://github.com/rust-lang/rust-clippy/pull/7795)
* Avoid [`eq_op`] in test functions
  [#7811](https://github.com/rust-lang/rust-clippy/pull/7811)
* [`cast_possible_truncation`] no longer lints when cast is coming from `signum`
  method call [#7850](https://github.com/rust-lang/rust-clippy/pull/7850)
* [`match_str_case_mismatch`] no longer lints on uncased characters
  [#7865](https://github.com/rust-lang/rust-clippy/pull/7865)
* [`ptr_arg`] no longer lints references to type aliases
  [#7890](https://github.com/rust-lang/rust-clippy/pull/7890)
* [`missing_safety_doc`] now also accepts "implementation safety" headers
  [#7856](https://github.com/rust-lang/rust-clippy/pull/7856)
* [`missing_safety_doc`] no longer lints if any parent has `#[doc(hidden)]`
  attribute [#7849](https://github.com/rust-lang/rust-clippy/pull/7849)
* [`if_not_else`] now ignores else-if statements
  [#7895](https://github.com/rust-lang/rust-clippy/pull/7895)
* Avoid linting [`cast_possible_truncation`] on bit-reducing operations
  [#7819](https://github.com/rust-lang/rust-clippy/pull/7819)
* Avoid linting [`field_reassign_with_default`] when `Drop` and `Copy` are
  involved [#7794](https://github.com/rust-lang/rust-clippy/pull/7794)
* [`unnecessary_sort_by`] now checks if argument implements `Ord` trait
  [#7824](https://github.com/rust-lang/rust-clippy/pull/7824)
* Fix false positive in [`match_overlapping_arm`]
  [#7847](https://github.com/rust-lang/rust-clippy/pull/7847)
* Prevent [`needless_lifetimes`] false positive in `async` function definition
  [#7901](https://github.com/rust-lang/rust-clippy/pull/7901)

### Suggestion fixes/improvements

* Keep an initial `::` when [`doc_markdown`] suggests to use ticks
  [#7916](https://github.com/rust-lang/rust-clippy/pull/7916)
* Add a machine applicable suggestion for the [`doc_markdown`] missing backticks
  lint [#7904](https://github.com/rust-lang/rust-clippy/pull/7904)
* [`equatable_if_let`] no longer expands macros in the suggestion
  [#7788](https://github.com/rust-lang/rust-clippy/pull/7788)
* Make [`shadow_reuse`] suggestion less verbose
  [#7782](https://github.com/rust-lang/rust-clippy/pull/7782)

### ICE fixes

* Fix ICE in [`enum_variant_names`]
  [#7873](https://github.com/rust-lang/rust-clippy/pull/7873)
* Fix ICE in [`undocumented_unsafe_blocks`]
  [#7891](https://github.com/rust-lang/rust-clippy/pull/7891)

### Documentation improvements

* Fixed naive doc formatting for `#[must_use]` lints ([`must_use_unit`],
  [`double_must_use`], [`must_use_candidate`], [`let_underscore_must_use`])
  [#7827](https://github.com/rust-lang/rust-clippy/pull/7827)
* Fix typo in example for [`match_result_ok`]
  [#7815](https://github.com/rust-lang/rust-clippy/pull/7815)

### Others

* Allow giving reasons for [`disallowed_types`]
  [#7791](https://github.com/rust-lang/rust-clippy/pull/7791)
* Fix [`manual_assert`] and [`match_wild_err_arm`] for `#![no_std]` and Rust
  2021. [#7851](https://github.com/rust-lang/rust-clippy/pull/7851)
* Fix regression in [`semicolon_if_nothing_returned`] on macros containing while
  loops [#7789](https://github.com/rust-lang/rust-clippy/pull/7789)
* Added a new configuration `literal-suffix-style` to enforce a certain style
  writing [`unseparated_literal_suffix`]
  [#7726](https://github.com/rust-lang/rust-clippy/pull/7726)

## Rust 1.57

Released 2021-12-02

[View all 148 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2021-08-12T20%3A36%3A04Z..2021-11-03T17%3A57%3A59Z+base%3Amaster)

### New Lints

* [`negative_feature_names`]
  [#7539](https://github.com/rust-lang/rust-clippy/pull/7539)
* [`redundant_feature_names`]
  [#7539](https://github.com/rust-lang/rust-clippy/pull/7539)
* [`mod_module_files`]
  [#7543](https://github.com/rust-lang/rust-clippy/pull/7543)
* [`self_named_module_files`]
  [#7543](https://github.com/rust-lang/rust-clippy/pull/7543)
* [`manual_split_once`]
  [#7565](https://github.com/rust-lang/rust-clippy/pull/7565)
* [`derivable_impls`]
  [#7570](https://github.com/rust-lang/rust-clippy/pull/7570)
* [`needless_option_as_deref`]
  [#7596](https://github.com/rust-lang/rust-clippy/pull/7596)
* [`iter_not_returning_iterator`]
  [#7610](https://github.com/rust-lang/rust-clippy/pull/7610)
* [`same_name_method`]
  [#7653](https://github.com/rust-lang/rust-clippy/pull/7653)
* [`manual_assert`] [#7669](https://github.com/rust-lang/rust-clippy/pull/7669)
* [`non_send_fields_in_send_ty`]
  [#7709](https://github.com/rust-lang/rust-clippy/pull/7709)
* [`equatable_if_let`]
  [#7762](https://github.com/rust-lang/rust-clippy/pull/7762)

### Moves and Deprecations

* Move [`shadow_unrelated`] to `restriction`
  [#7338](https://github.com/rust-lang/rust-clippy/pull/7338)
* Move [`option_if_let_else`] to `nursery`
  [#7568](https://github.com/rust-lang/rust-clippy/pull/7568)
* Move [`branches_sharing_code`] to `nursery`
  [#7595](https://github.com/rust-lang/rust-clippy/pull/7595)
* Rename `if_let_some_result` to [`match_result_ok`] which now also handles
  `while let` cases [#7608](https://github.com/rust-lang/rust-clippy/pull/7608)
* Move [`many_single_char_names`] to `pedantic`
  [#7671](https://github.com/rust-lang/rust-clippy/pull/7671)
* Move [`float_cmp`] to `pedantic`
  [#7692](https://github.com/rust-lang/rust-clippy/pull/7692)
* Rename `box_vec` to [`box_collection`] and lint on more general cases
  [#7693](https://github.com/rust-lang/rust-clippy/pull/7693)
* Uplift `invalid_atomic_ordering` to rustc
  [rust-lang/rust#84039](https://github.com/rust-lang/rust/pull/84039)

### Enhancements

* Rewrite the `shadow*` lints, so that they find a lot more shadows and are not
  limited to certain patterns
  [#7338](https://github.com/rust-lang/rust-clippy/pull/7338)
* The `avoid-breaking-exported-api` configuration now also works for
  [`box_collection`], [`redundant_allocation`], [`rc_buffer`], [`vec_box`],
  [`option_option`], [`linkedlist`], [`rc_mutex`]
  [#7560](https://github.com/rust-lang/rust-clippy/pull/7560)
* [`unnecessary_unwrap`]: Now also checks for `expect`s
  [#7584](https://github.com/rust-lang/rust-clippy/pull/7584)
* [`disallowed_methods`]: Allow adding a reason that will be displayed with the
  lint message
  [#7621](https://github.com/rust-lang/rust-clippy/pull/7621)
* [`approx_constant`]: Now checks the MSRV for `LOG10_2` and `LOG2_10`
  [#7629](https://github.com/rust-lang/rust-clippy/pull/7629)
* [`approx_constant`]: Add `TAU`
  [#7642](https://github.com/rust-lang/rust-clippy/pull/7642)
* [`needless_borrow`]: Now also lints on needless mutable borrows
  [#7657](https://github.com/rust-lang/rust-clippy/pull/7657)
* [`missing_safety_doc`]: Now also lints on unsafe traits
  [#7734](https://github.com/rust-lang/rust-clippy/pull/7734)

### False Positive Fixes

* [`manual_map`]: No longer lints when the option is borrowed in the match and
  also consumed in the arm
  [#7531](https://github.com/rust-lang/rust-clippy/pull/7531)
* [`filter_next`]: No longer lints if `filter` method is not the
  `Iterator::filter` method
  [#7562](https://github.com/rust-lang/rust-clippy/pull/7562)
* [`manual_flatten`]: No longer lints if expression is used after `if let`
  [#7566](https://github.com/rust-lang/rust-clippy/pull/7566)
* [`option_if_let_else`]: Multiple fixes
  [#7573](https://github.com/rust-lang/rust-clippy/pull/7573)
    * `break` and `continue` statements local to the would-be closure are
      allowed
    * Don't lint in const contexts
    * Don't lint when yield expressions are used
    * Don't lint when the captures made by the would-be closure conflict with
      the other branch
    * Don't lint when a field of a local is used when the type could be
      potentially moved from
    * In some cases, don't lint when scrutinee expression conflicts with the
      captures of the would-be closure
* [`redundant_allocation`]: No longer lints on `Box<Box<dyn T>>` which replaces
  wide pointers with thin pointers
  [#7592](https://github.com/rust-lang/rust-clippy/pull/7592)
* [`bool_assert_comparison`]: No longer lints on types that do not implement the
  `Not` trait with `Output = bool`
  [#7605](https://github.com/rust-lang/rust-clippy/pull/7605)
* [`mut_range_bound`]: No longer lints on range bound mutations, that are
  immediately followed by a `break;`
  [#7607](https://github.com/rust-lang/rust-clippy/pull/7607)
* [`mutable_key_type`]: Improve accuracy and document remaining false positives
  and false negatives
  [#7640](https://github.com/rust-lang/rust-clippy/pull/7640)
* [`redundant_closure`]: Rewrite the lint to fix various false positives and
  false negatives [#7661](https://github.com/rust-lang/rust-clippy/pull/7661)
* [`large_enum_variant`]: No longer wrongly identifies the second largest
  variant [#7677](https://github.com/rust-lang/rust-clippy/pull/7677)
* [`needless_return`]: No longer lints on let-else expressions
  [#7685](https://github.com/rust-lang/rust-clippy/pull/7685)
* [`suspicious_else_formatting`]: No longer lints in proc-macros
  [#7707](https://github.com/rust-lang/rust-clippy/pull/7707)
* [`excessive_precision`]: No longer lints when in some cases the float was
  already written in the shortest form
  [#7722](https://github.com/rust-lang/rust-clippy/pull/7722)
* [`doc_markdown`]: No longer lints on intra-doc links
  [#7772](https://github.com/rust-lang/rust-clippy/pull/7772)

### Suggestion Fixes/Improvements

* [`unnecessary_operation`]: Recommend using an `assert!` instead of using a
  function call in an indexing operation
  [#7453](https://github.com/rust-lang/rust-clippy/pull/7453)
* [`manual_split_once`]: Produce semantically equivalent suggestion when
  `rsplitn` is used [#7663](https://github.com/rust-lang/rust-clippy/pull/7663)
* [`while_let_on_iterator`]: Produce correct suggestion when using `&mut`
  [#7690](https://github.com/rust-lang/rust-clippy/pull/7690)
* [`manual_assert`]: No better handles complex conditions
  [#7741](https://github.com/rust-lang/rust-clippy/pull/7741)
* Correctly handle signs in exponents in numeric literals lints
  [#7747](https://github.com/rust-lang/rust-clippy/pull/7747)
* [`suspicious_map`]: Now also suggests to use `inspect` as an alternative
  [#7770](https://github.com/rust-lang/rust-clippy/pull/7770)
* Drop exponent from suggestion if it is 0 in numeric literals lints
  [#7774](https://github.com/rust-lang/rust-clippy/pull/7774)

### ICE Fixes

* [`implicit_hasher`]
  [#7761](https://github.com/rust-lang/rust-clippy/pull/7761)

### Others

* Clippy now uses the 2021
  [Edition!](https://www.youtube.com/watch?v=q0aNduqb2Ro)
  [#7664](https://github.com/rust-lang/rust-clippy/pull/7664)

## Rust 1.56

Released 2021-10-21

[View all 38 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2021-07-19T14%3A33%3A33Z..2021-08-12T09%3A28%3A38Z+base%3Amaster)

### New Lints

* [`unwrap_or_else_default`]
  [#7516](https://github.com/rust-lang/rust-clippy/pull/7516)

### Enhancements

* [`needless_continue`]: Now also lints in `loop { continue; }` case
  [#7477](https://github.com/rust-lang/rust-clippy/pull/7477)
* [`disallowed_types`]: Now also primitive types can be disallowed
  [#7488](https://github.com/rust-lang/rust-clippy/pull/7488)
* [`manual_swap`]: Now also lints on xor swaps
  [#7506](https://github.com/rust-lang/rust-clippy/pull/7506)
* [`map_flatten`]: Now also lints on the `Result` type
  [#7522](https://github.com/rust-lang/rust-clippy/pull/7522)
* [`no_effect`]: Now also lints on inclusive ranges
  [#7556](https://github.com/rust-lang/rust-clippy/pull/7556)

### False Positive Fixes

* [`nonstandard_macro_braces`]: No longer lints on similar named nested macros
  [#7478](https://github.com/rust-lang/rust-clippy/pull/7478)
* [`too_many_lines`]: No longer lints in closures to avoid duplicated diagnostics
  [#7534](https://github.com/rust-lang/rust-clippy/pull/7534)
* [`similar_names`]: No longer complains about `iter` and `item` being too
  similar [#7546](https://github.com/rust-lang/rust-clippy/pull/7546)

### Suggestion Fixes/Improvements

* [`similar_names`]: No longer suggests to insert or add an underscore as a fix
  [#7221](https://github.com/rust-lang/rust-clippy/pull/7221)
* [`new_without_default`]: No longer shows the full qualified type path when
  suggesting adding a `Default` implementation
  [#7493](https://github.com/rust-lang/rust-clippy/pull/7493)
* [`while_let_on_iterator`]: Now suggests re-borrowing mutable references
  [#7520](https://github.com/rust-lang/rust-clippy/pull/7520)
* [`extend_with_drain`]: Improve code suggestion for mutable and immutable
  references [#7533](https://github.com/rust-lang/rust-clippy/pull/7533)
* [`trivially_copy_pass_by_ref`]: Now properly handles `Self` type
  [#7535](https://github.com/rust-lang/rust-clippy/pull/7535)
* [`never_loop`]: Now suggests using `if let` instead of a `for` loop when
  applicable [#7541](https://github.com/rust-lang/rust-clippy/pull/7541)

### Documentation Improvements

* Clippy now uses a lint to generate its lint documentation. [Lints all the way
  down](https://en.wikipedia.org/wiki/Turtles_all_the_way_down).
  [#7502](https://github.com/rust-lang/rust-clippy/pull/7502)
* Reworked Clippy's website:
  [#7172](https://github.com/rust-lang/rust-clippy/issues/7172)
  [#7279](https://github.com/rust-lang/rust-clippy/pull/7279)
  * Added applicability information about lints
  * Added a link to jump into the implementation
  * Improved loading times
  * Adapted some styling
* `cargo clippy --help` now also explains the `--fix` and `--no-deps` flag
  [#7492](https://github.com/rust-lang/rust-clippy/pull/7492)
* [`unnested_or_patterns`]: Removed `or_patterns` feature gate in the code
  example [#7507](https://github.com/rust-lang/rust-clippy/pull/7507)

## Rust 1.55

Released 2021-09-09

[View all 83 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2021-06-03T07%3A23%3A59Z..2021-07-29T11%3A47%3A32Z+base%3Amaster)

### Important Changes

* Stabilized `cargo clippy --fix` :tada:
  [#7405](https://github.com/rust-lang/rust-clippy/pull/7405)

### New Lints

* [`rc_mutex`]
  [#7316](https://github.com/rust-lang/rust-clippy/pull/7316)
* [`nonstandard_macro_braces`]
  [#7299](https://github.com/rust-lang/rust-clippy/pull/7299)
* [`strlen_on_c_strings`]
  [#7243](https://github.com/rust-lang/rust-clippy/pull/7243)
* [`self_named_constructors`]
  [#7403](https://github.com/rust-lang/rust-clippy/pull/7403)
* [`disallowed_script_idents`]
  [#7400](https://github.com/rust-lang/rust-clippy/pull/7400)
* [`disallowed_types`]
  [#7315](https://github.com/rust-lang/rust-clippy/pull/7315)
* [`missing_enforced_import_renames`]
  [#7300](https://github.com/rust-lang/rust-clippy/pull/7300)
* [`extend_with_drain`]
  [#7270](https://github.com/rust-lang/rust-clippy/pull/7270)

### Moves and Deprecations

* Moved [`from_iter_instead_of_collect`] to `pedantic`
  [#7375](https://github.com/rust-lang/rust-clippy/pull/7375)
* Added `suspicious` as a new lint group for *code that is most likely wrong or useless*
  [#7350](https://github.com/rust-lang/rust-clippy/pull/7350)
  * Moved [`blanket_clippy_restriction_lints`] to `suspicious`
  * Moved [`empty_loop`] to `suspicious`
  * Moved [`eval_order_dependence`] to `suspicious`
  * Moved [`float_equality_without_abs`] to `suspicious`
  * Moved [`for_loops_over_fallibles`] to `suspicious`
  * Moved [`misrefactored_assign_op`] to `suspicious`
  * Moved [`mut_range_bound`] to `suspicious`
  * Moved [`mutable_key_type`] to `suspicious`
  * Moved [`suspicious_arithmetic_impl`] to `suspicious`
  * Moved [`suspicious_assignment_formatting`] to `suspicious`
  * Moved [`suspicious_else_formatting`] to `suspicious`
  * Moved [`suspicious_map`] to `suspicious`
  * Moved [`suspicious_op_assign_impl`] to `suspicious`
  * Moved [`suspicious_unary_op_formatting`] to `suspicious`

### Enhancements

* [`while_let_on_iterator`]: Now suggests `&mut iter` inside closures
  [#7262](https://github.com/rust-lang/rust-clippy/pull/7262)
* [`doc_markdown`]:
  * Now detects unbalanced ticks
    [#7357](https://github.com/rust-lang/rust-clippy/pull/7357)
  * Add `FreeBSD` to the default configuration as an allowed identifier
    [#7334](https://github.com/rust-lang/rust-clippy/pull/7334)
* [`wildcard_enum_match_arm`], [`match_wildcard_for_single_variants`]: Now allows wildcards for enums with unstable
  or hidden variants
  [#7407](https://github.com/rust-lang/rust-clippy/pull/7407)
* [`redundant_allocation`]: Now additionally supports the `Arc<>` type
  [#7308](https://github.com/rust-lang/rust-clippy/pull/7308)
* [`disallowed_names`]: Now allows disallowed names in test code
  [#7379](https://github.com/rust-lang/rust-clippy/pull/7379)
* [`redundant_closure`]: Suggests `&mut` for `FnMut`
  [#7437](https://github.com/rust-lang/rust-clippy/pull/7437)
* [`disallowed_methods`], [`disallowed_types`]: The configuration values `disallowed-method` and `disallowed-type`
  no longer require fully qualified paths
  [#7345](https://github.com/rust-lang/rust-clippy/pull/7345)
* [`zst_offset`]: Fixed lint invocation after it was accidentally suppressed
  [#7396](https://github.com/rust-lang/rust-clippy/pull/7396)

### False Positive Fixes

* [`default_numeric_fallback`]: No longer lints on float literals as function arguments
  [#7446](https://github.com/rust-lang/rust-clippy/pull/7446)
* [`use_self`]: No longer lints on type parameters
  [#7288](https://github.com/rust-lang/rust-clippy/pull/7288)
* [`unimplemented`]: Now ignores the `assert` and `debug_assert` macros
  [#7439](https://github.com/rust-lang/rust-clippy/pull/7439)
* [`branches_sharing_code`]: Now always checks for block expressions
  [#7462](https://github.com/rust-lang/rust-clippy/pull/7462)
* [`field_reassign_with_default`]: No longer triggers in macros
  [#7160](https://github.com/rust-lang/rust-clippy/pull/7160)
* [`redundant_clone`]: No longer lints on required clones for borrowed data
  [#7346](https://github.com/rust-lang/rust-clippy/pull/7346)
* [`default_numeric_fallback`]: No longer triggers in external macros
  [#7325](https://github.com/rust-lang/rust-clippy/pull/7325)
* [`needless_bool`]: No longer lints in macros
  [#7442](https://github.com/rust-lang/rust-clippy/pull/7442)
* [`useless_format`]: No longer triggers when additional text is being appended
  [#7442](https://github.com/rust-lang/rust-clippy/pull/7442)
* [`assertions_on_constants`]: `cfg!(...)` is no longer considered to be a constant
  [#7319](https://github.com/rust-lang/rust-clippy/pull/7319)

### Suggestion Fixes/Improvements

* [`needless_collect`]: Now show correct lint messages for shadowed values
  [#7289](https://github.com/rust-lang/rust-clippy/pull/7289)
* [`wrong_pub_self_convention`]: The deprecated message now suggest the correct configuration value
  [#7382](https://github.com/rust-lang/rust-clippy/pull/7382)
* [`semicolon_if_nothing_returned`]: Allow missing semicolon in blocks with only one expression
  [#7326](https://github.com/rust-lang/rust-clippy/pull/7326)

### ICE Fixes

* [`zero_sized_map_values`]
  [#7470](https://github.com/rust-lang/rust-clippy/pull/7470)
* [`redundant_pattern_matching`]
  [#7471](https://github.com/rust-lang/rust-clippy/pull/7471)
* [`modulo_one`]
  [#7473](https://github.com/rust-lang/rust-clippy/pull/7473)
* [`use_self`]
  [#7428](https://github.com/rust-lang/rust-clippy/pull/7428)

## Rust 1.54

Released 2021-07-29

[View all 74 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2021-04-27T23%3A51%3A18Z..2021-06-03T06%3A54%3A07Z+base%3Amaster)

### New Lints

- [`ref_binding_to_reference`]
  [#7105](https://github.com/rust-lang/rust-clippy/pull/7105)
- [`needless_bitwise_bool`]
  [#7133](https://github.com/rust-lang/rust-clippy/pull/7133)
- [`unused_async`] [#7225](https://github.com/rust-lang/rust-clippy/pull/7225)
- [`manual_str_repeat`]
  [#7265](https://github.com/rust-lang/rust-clippy/pull/7265)
- [`suspicious_splitn`]
  [#7292](https://github.com/rust-lang/rust-clippy/pull/7292)

### Moves and Deprecations

- Deprecate `pub_enum_variant_names` and `wrong_pub_self_convention` in favor of
  the new `avoid-breaking-exported-api` config option (see
  [Enhancements](#1-54-enhancements))
  [#7187](https://github.com/rust-lang/rust-clippy/pull/7187)
- Move [`inconsistent_struct_constructor`] to `pedantic`
  [#7193](https://github.com/rust-lang/rust-clippy/pull/7193)
- Move [`needless_borrow`] to `style` (now warn-by-default)
  [#7254](https://github.com/rust-lang/rust-clippy/pull/7254)
- Move [`suspicious_operation_groupings`] to `nursery`
  [#7266](https://github.com/rust-lang/rust-clippy/pull/7266)
- Move [`semicolon_if_nothing_returned`] to `pedantic`
  [#7268](https://github.com/rust-lang/rust-clippy/pull/7268)

### Enhancements <a name="1-54-enhancements"></a>

- [`while_let_on_iterator`]: Now also lints in nested loops
  [#6966](https://github.com/rust-lang/rust-clippy/pull/6966)
- [`single_char_pattern`]: Now also lints on `strip_prefix` and `strip_suffix`
  [#7156](https://github.com/rust-lang/rust-clippy/pull/7156)
- [`needless_collect`]: Now also lints on assignments with type annotations
  [#7163](https://github.com/rust-lang/rust-clippy/pull/7163)
- [`if_then_some_else_none`]: Now works with the MSRV config
  [#7177](https://github.com/rust-lang/rust-clippy/pull/7177)
- Add `avoid-breaking-exported-api` config option for the lints
  [`enum_variant_names`], [`large_types_passed_by_value`],
  [`trivially_copy_pass_by_ref`], [`unnecessary_wraps`],
  [`upper_case_acronyms`], and [`wrong_self_convention`]. We recommend to set
  this configuration option to `false` before a major release (1.0/2.0/...) to
  clean up the API [#7187](https://github.com/rust-lang/rust-clippy/pull/7187)
- [`needless_collect`]: Now lints on even more data structures
  [#7188](https://github.com/rust-lang/rust-clippy/pull/7188)
- [`missing_docs_in_private_items`]: No longer sees `#[<name> = "<value>"]` like
  attributes as sufficient documentation
  [#7281](https://github.com/rust-lang/rust-clippy/pull/7281)
- [`needless_collect`], [`short_circuit_statement`], [`unnecessary_operation`]:
  Now work as expected when used with `allow`
  [#7282](https://github.com/rust-lang/rust-clippy/pull/7282)

### False Positive Fixes

- [`implicit_return`]: Now takes all diverging functions in account to avoid
  false positives [#6951](https://github.com/rust-lang/rust-clippy/pull/6951)
- [`while_let_on_iterator`]: No longer lints when the iterator is a struct field
  and the struct is used in the loop
  [#6966](https://github.com/rust-lang/rust-clippy/pull/6966)
- [`multiple_inherent_impl`]: No longer lints with generic arguments
  [#7089](https://github.com/rust-lang/rust-clippy/pull/7089)
- [`comparison_chain`]: No longer lints in a `const` context
  [#7118](https://github.com/rust-lang/rust-clippy/pull/7118)
- [`while_immutable_condition`]: Fix false positive where mutation in the loop
  variable wasn't picked up
  [#7144](https://github.com/rust-lang/rust-clippy/pull/7144)
- [`default_trait_access`]: No longer lints in macros
  [#7150](https://github.com/rust-lang/rust-clippy/pull/7150)
- [`needless_question_mark`]: No longer lints when the inner value is implicitly
  dereferenced [#7165](https://github.com/rust-lang/rust-clippy/pull/7165)
- [`unused_unit`]: No longer lints when multiple macro contexts are involved
  [#7167](https://github.com/rust-lang/rust-clippy/pull/7167)
- [`eval_order_dependence`]: Fix false positive in async context
  [#7174](https://github.com/rust-lang/rust-clippy/pull/7174)
- [`unnecessary_filter_map`]: No longer lints if the `filter_map` changes the
  type [#7175](https://github.com/rust-lang/rust-clippy/pull/7175)
- [`wrong_self_convention`]: No longer lints in trait implementations of
  non-`Copy` types [#7182](https://github.com/rust-lang/rust-clippy/pull/7182)
- [`suboptimal_flops`]: No longer lints on `powi(2)`
  [#7201](https://github.com/rust-lang/rust-clippy/pull/7201)
- [`wrong_self_convention`]: No longer lints if there is no implicit `self`
  [#7215](https://github.com/rust-lang/rust-clippy/pull/7215)
- [`option_if_let_else`]: No longer lints on `else if let` pattern
  [#7216](https://github.com/rust-lang/rust-clippy/pull/7216)
- [`use_self`], [`useless_conversion`]: Fix false positives when generic
  arguments are involved
  [#7223](https://github.com/rust-lang/rust-clippy/pull/7223)
- [`manual_unwrap_or`]: Fix false positive with deref coercion
  [#7233](https://github.com/rust-lang/rust-clippy/pull/7233)
- [`similar_names`]: No longer lints on `wparam`/`lparam`
  [#7255](https://github.com/rust-lang/rust-clippy/pull/7255)
- [`redundant_closure`]: No longer lints on using the `vec![]` macro in a
  closure [#7263](https://github.com/rust-lang/rust-clippy/pull/7263)

### Suggestion Fixes/Improvements

- [`implicit_return`]
  [#6951](https://github.com/rust-lang/rust-clippy/pull/6951)
    - Fix suggestion for async functions
    - Improve suggestion with macros
    - Suggest to change `break` to `return` when appropriate
- [`while_let_on_iterator`]: Now suggests `&mut iter` when necessary
  [#6966](https://github.com/rust-lang/rust-clippy/pull/6966)
- [`match_single_binding`]: Improve suggestion when match scrutinee has side
  effects [#7095](https://github.com/rust-lang/rust-clippy/pull/7095)
- [`needless_borrow`]: Now suggests to also change usage sites as needed
  [#7105](https://github.com/rust-lang/rust-clippy/pull/7105)
- [`write_with_newline`]: Improve suggestion when only `\n` is written to the
  buffer [#7183](https://github.com/rust-lang/rust-clippy/pull/7183)
- [`from_iter_instead_of_collect`]: The suggestion is now auto applicable also
  when a `<_ as Trait>::_` is involved
  [#7264](https://github.com/rust-lang/rust-clippy/pull/7264)
- [`not_unsafe_ptr_arg_deref`]: Improved error message
  [#7294](https://github.com/rust-lang/rust-clippy/pull/7294)

### ICE Fixes

- Fix ICE when running Clippy on `libstd`
  [#7140](https://github.com/rust-lang/rust-clippy/pull/7140)
- [`implicit_return`]
  [#7242](https://github.com/rust-lang/rust-clippy/pull/7242)

## Rust 1.53

Released 2021-06-17

[View all 126 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2021-03-12T22%3A49%3A20Z..2021-04-27T14%3A38%3A20Z+base%3Amaster)

### New Lints

* [`option_filter_map`]
  [#6342](https://github.com/rust-lang/rust-clippy/pull/6342)
* [`branches_sharing_code`]
  [#6463](https://github.com/rust-lang/rust-clippy/pull/6463)
* [`needless_for_each`]
  [#6706](https://github.com/rust-lang/rust-clippy/pull/6706)
* [`if_then_some_else_none`]
  [#6859](https://github.com/rust-lang/rust-clippy/pull/6859)
* [`non_octal_unix_permissions`]
  [#7001](https://github.com/rust-lang/rust-clippy/pull/7001)
* [`unnecessary_self_imports`]
  [#7072](https://github.com/rust-lang/rust-clippy/pull/7072)
* [`bool_assert_comparison`]
  [#7083](https://github.com/rust-lang/rust-clippy/pull/7083)
* [`cloned_instead_of_copied`]
  [#7098](https://github.com/rust-lang/rust-clippy/pull/7098)
* [`flat_map_option`]
  [#7101](https://github.com/rust-lang/rust-clippy/pull/7101)

### Moves and Deprecations

* Deprecate [`filter_map`] lint
  [#7059](https://github.com/rust-lang/rust-clippy/pull/7059)
* Move [`transmute_ptr_to_ptr`] to `pedantic`
  [#7102](https://github.com/rust-lang/rust-clippy/pull/7102)

### Enhancements

* [`mem_replace_with_default`]: Also lint on common std constructors
  [#6820](https://github.com/rust-lang/rust-clippy/pull/6820)
* [`wrong_self_convention`]: Also lint on `to_*_mut` methods
  [#6828](https://github.com/rust-lang/rust-clippy/pull/6828)
* [`wildcard_enum_match_arm`], [`match_wildcard_for_single_variants`]:
  [#6863](https://github.com/rust-lang/rust-clippy/pull/6863)
    * Attempt to find a common path prefix in suggestion
    * Don't lint on `Option` and `Result`
    * Consider `Self` prefix
* [`explicit_deref_methods`]: Also lint on chained `deref` calls
  [#6865](https://github.com/rust-lang/rust-clippy/pull/6865)
* [`or_fun_call`]: Also lint on `unsafe` blocks
  [#6928](https://github.com/rust-lang/rust-clippy/pull/6928)
* [`vec_box`], [`linkedlist`], [`option_option`]: Also lint in `const` and
  `static` items [#6938](https://github.com/rust-lang/rust-clippy/pull/6938)
* [`search_is_some`]: Also check for `is_none`
  [#6942](https://github.com/rust-lang/rust-clippy/pull/6942)
* [`string_lit_as_bytes`]: Also lint on `into_bytes`
  [#6959](https://github.com/rust-lang/rust-clippy/pull/6959)
* [`len_without_is_empty`]: Also lint if function signatures of `len` and
  `is_empty` don't match
  [#6980](https://github.com/rust-lang/rust-clippy/pull/6980)
* [`redundant_pattern_matching`]: Also lint if the pattern is a `&` pattern
  [#6991](https://github.com/rust-lang/rust-clippy/pull/6991)
* [`clone_on_copy`]: Also lint on chained method calls taking `self` by value
  [#7000](https://github.com/rust-lang/rust-clippy/pull/7000)
* [`missing_panics_doc`]: Also lint on `assert_eq!` and `assert_ne!`
  [#7029](https://github.com/rust-lang/rust-clippy/pull/7029)
* [`needless_return`]: Also lint in `async` functions
  [#7067](https://github.com/rust-lang/rust-clippy/pull/7067)
* [`unused_io_amount`]: Also lint on expressions like `_.read().ok()?`
  [#7100](https://github.com/rust-lang/rust-clippy/pull/7100)
* [`iter_cloned_collect`]: Also lint on large arrays, since const-generics are
  now stable [#7138](https://github.com/rust-lang/rust-clippy/pull/7138)

### False Positive Fixes

* [`upper_case_acronyms`]: No longer lints on public items
  [#6805](https://github.com/rust-lang/rust-clippy/pull/6805)
* [`suspicious_map`]: No longer lints when side effects may occur inside the
  `map` call [#6831](https://github.com/rust-lang/rust-clippy/pull/6831)
* [`manual_map`], [`manual_unwrap_or`]: No longer lints in `const` functions
  [#6917](https://github.com/rust-lang/rust-clippy/pull/6917)
* [`wrong_self_convention`]: Now respects `Copy` types
  [#6924](https://github.com/rust-lang/rust-clippy/pull/6924)
* [`needless_question_mark`]: No longer lints if the `?` and the `Some(..)` come
  from different macro contexts [#6935](https://github.com/rust-lang/rust-clippy/pull/6935)
* [`map_entry`]: Better detect if the entry API can be used
  [#6937](https://github.com/rust-lang/rust-clippy/pull/6937)
* [`or_fun_call`]: No longer lints on some `len` function calls
  [#6950](https://github.com/rust-lang/rust-clippy/pull/6950)
* [`new_ret_no_self`]: No longer lints when `Self` is returned with different
  generic arguments [#6952](https://github.com/rust-lang/rust-clippy/pull/6952)
* [`upper_case_acronyms`]: No longer lints on public items
  [#6981](https://github.com/rust-lang/rust-clippy/pull/6981)
* [`explicit_into_iter_loop`]: Only lint when `into_iter` is an implementation
  of `IntoIterator` [#6982](https://github.com/rust-lang/rust-clippy/pull/6982)
* [`expl_impl_clone_on_copy`]: Take generic constraints into account before
  suggesting to use `derive` instead
  [#6993](https://github.com/rust-lang/rust-clippy/pull/6993)
* [`missing_panics_doc`]: No longer lints when only debug-assertions are used
  [#6996](https://github.com/rust-lang/rust-clippy/pull/6996)
* [`clone_on_copy`]: Only lint when using the `Clone` trait
  [#7000](https://github.com/rust-lang/rust-clippy/pull/7000)
* [`wrong_self_convention`]: No longer lints inside a trait implementation
  [#7002](https://github.com/rust-lang/rust-clippy/pull/7002)
* [`redundant_clone`]: No longer lints when the cloned value is modified while
  the clone is in use
  [#7011](https://github.com/rust-lang/rust-clippy/pull/7011)
* [`same_item_push`]: No longer lints if the `Vec` is used in the loop body
  [#7018](https://github.com/rust-lang/rust-clippy/pull/7018)
* [`cargo_common_metadata`]: Remove author requirement
  [#7026](https://github.com/rust-lang/rust-clippy/pull/7026)
* [`panic_in_result_fn`]: No longer lints on `debug_assert` family
  [#7060](https://github.com/rust-lang/rust-clippy/pull/7060)
* [`panic`]: No longer wrongfully lints on `debug_assert` with message
  [#7063](https://github.com/rust-lang/rust-clippy/pull/7063)
* [`wrong_self_convention`]: No longer lints in trait implementations where no
  `self` is involved [#7064](https://github.com/rust-lang/rust-clippy/pull/7064)
* [`missing_const_for_fn`]: No longer lints when unstable `const` function is
  involved [#7076](https://github.com/rust-lang/rust-clippy/pull/7076)
* [`suspicious_else_formatting`]: Allow Allman style braces
  [#7087](https://github.com/rust-lang/rust-clippy/pull/7087)
* [`inconsistent_struct_constructor`]: No longer lints in macros
  [#7097](https://github.com/rust-lang/rust-clippy/pull/7097)
* [`single_component_path_imports`]: No longer lints on macro re-exports
  [#7120](https://github.com/rust-lang/rust-clippy/pull/7120)

### Suggestion Fixes/Improvements

* [`redundant_pattern_matching`]: Add a note when applying this lint would
  change the drop order
  [#6568](https://github.com/rust-lang/rust-clippy/pull/6568)
* [`write_literal`], [`print_literal`]: Add auto-applicable suggestion
  [#6821](https://github.com/rust-lang/rust-clippy/pull/6821)
* [`manual_map`]: Fix suggestion for complex `if let ... else` chains
  [#6856](https://github.com/rust-lang/rust-clippy/pull/6856)
* [`inconsistent_struct_constructor`]: Make lint description and message clearer
  [#6892](https://github.com/rust-lang/rust-clippy/pull/6892)
* [`map_entry`]: Now suggests `or_insert`, `insert_with` or `match _.entry(_)`
  as appropriate [#6937](https://github.com/rust-lang/rust-clippy/pull/6937)
* [`manual_flatten`]: Suggest to insert `copied` if necessary
  [#6962](https://github.com/rust-lang/rust-clippy/pull/6962)
* [`redundant_slicing`]: Fix suggestion when a re-borrow might be required or
  when the value is from a macro call
  [#6975](https://github.com/rust-lang/rust-clippy/pull/6975)
* [`match_wildcard_for_single_variants`]: Fix suggestion for hidden variant
  [#6988](https://github.com/rust-lang/rust-clippy/pull/6988)
* [`clone_on_copy`]: Correct suggestion when the cloned value is a macro call
  [#7000](https://github.com/rust-lang/rust-clippy/pull/7000)
* [`manual_map`]: Fix suggestion at the end of an if chain
  [#7004](https://github.com/rust-lang/rust-clippy/pull/7004)
* Fix needless parenthesis output in multiple lint suggestions
  [#7013](https://github.com/rust-lang/rust-clippy/pull/7013)
* [`needless_collect`]: Better explanation in the lint message
  [#7020](https://github.com/rust-lang/rust-clippy/pull/7020)
* [`useless_vec`]: Now considers mutability
  [#7036](https://github.com/rust-lang/rust-clippy/pull/7036)
* [`useless_format`]: Wrap the content in braces if necessary
  [#7092](https://github.com/rust-lang/rust-clippy/pull/7092)
* [`single_match`]: Don't suggest an equality check for types which don't
  implement `PartialEq`
  [#7093](https://github.com/rust-lang/rust-clippy/pull/7093)
* [`from_over_into`]: Mention type in help message
  [#7099](https://github.com/rust-lang/rust-clippy/pull/7099)
* [`manual_unwrap_or`]: Fix invalid code suggestion due to a macro call
  [#7136](https://github.com/rust-lang/rust-clippy/pull/7136)

### ICE Fixes

* [`macro_use_imports`]
  [#7022](https://github.com/rust-lang/rust-clippy/pull/7022)
* [`missing_panics_doc`]
  [#7034](https://github.com/rust-lang/rust-clippy/pull/7034)
* [`tabs_in_doc_comments`]
  [#7039](https://github.com/rust-lang/rust-clippy/pull/7039)
* [`missing_const_for_fn`]
  [#7128](https://github.com/rust-lang/rust-clippy/pull/7128)

### Others

* [Clippy's lint
  list](https://rust-lang.github.io/rust-clippy/master/index.html) now supports
  themes [#7030](https://github.com/rust-lang/rust-clippy/pull/7030)
* Lints that were uplifted to `rustc` now mention the new `rustc` name in the
  deprecation warning
  [#7056](https://github.com/rust-lang/rust-clippy/pull/7056)

## Rust 1.52

Released 2021-05-06

[View all 102 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2021-02-03T15%3A59%3A06Z..2021-03-11T20%3A06%3A43Z+base%3Amaster)

### New Lints

* [`from_str_radix_10`]
  [#6717](https://github.com/rust-lang/rust-clippy/pull/6717)
* [`implicit_clone`]
  [#6730](https://github.com/rust-lang/rust-clippy/pull/6730)
* [`semicolon_if_nothing_returned`]
  [#6681](https://github.com/rust-lang/rust-clippy/pull/6681)
* [`manual_flatten`]
  [#6646](https://github.com/rust-lang/rust-clippy/pull/6646)
* [`inconsistent_struct_constructor`]
  [#6769](https://github.com/rust-lang/rust-clippy/pull/6769)
* [`iter_count`]
  [#6791](https://github.com/rust-lang/rust-clippy/pull/6791)
* [`default_numeric_fallback`]
  [#6662](https://github.com/rust-lang/rust-clippy/pull/6662)
* [`bytes_nth`]
  [#6695](https://github.com/rust-lang/rust-clippy/pull/6695)
* [`filter_map_identity`]
  [#6685](https://github.com/rust-lang/rust-clippy/pull/6685)
* [`manual_map`]
  [#6573](https://github.com/rust-lang/rust-clippy/pull/6573)

### Moves and Deprecations

* Moved [`upper_case_acronyms`] to `pedantic`
  [#6775](https://github.com/rust-lang/rust-clippy/pull/6775)
* Moved [`manual_map`] to `nursery`
  [#6796](https://github.com/rust-lang/rust-clippy/pull/6796)
* Moved [`unnecessary_wraps`] to `pedantic`
  [#6765](https://github.com/rust-lang/rust-clippy/pull/6765)
* Moved [`trivial_regex`] to `nursery`
  [#6696](https://github.com/rust-lang/rust-clippy/pull/6696)
* Moved [`naive_bytecount`] to `pedantic`
  [#6825](https://github.com/rust-lang/rust-clippy/pull/6825)
* Moved [`upper_case_acronyms`] to `style`
  [#6788](https://github.com/rust-lang/rust-clippy/pull/6788)
* Moved [`manual_map`] to `style`
  [#6801](https://github.com/rust-lang/rust-clippy/pull/6801)

### Enhancements

* [`disallowed_methods`]: Now supports functions in addition to methods
  [#6674](https://github.com/rust-lang/rust-clippy/pull/6674)
* [`upper_case_acronyms`]: Added a new configuration `upper-case-acronyms-aggressive` to
  trigger the lint if there is more than one uppercase character next to each other
  [#6788](https://github.com/rust-lang/rust-clippy/pull/6788)
* [`collapsible_match`]: Now supports block comparison with different value names
  [#6754](https://github.com/rust-lang/rust-clippy/pull/6754)
* [`unnecessary_wraps`]: Will now suggest removing unnecessary wrapped return unit type, like `Option<()>`
  [#6665](https://github.com/rust-lang/rust-clippy/pull/6665)
* Improved value usage detection in closures
  [#6698](https://github.com/rust-lang/rust-clippy/pull/6698)

### False Positive Fixes

* [`use_self`]: No longer lints in macros
  [#6833](https://github.com/rust-lang/rust-clippy/pull/6833)
* [`use_self`]: Fixed multiple false positives for: generics, associated types and derive implementations
  [#6179](https://github.com/rust-lang/rust-clippy/pull/6179)
* [`missing_inline_in_public_items`]: No longer lints for procedural macros
  [#6814](https://github.com/rust-lang/rust-clippy/pull/6814)
* [`inherent_to_string`]: No longer lints on functions with function generics
  [#6771](https://github.com/rust-lang/rust-clippy/pull/6771)
* [`doc_markdown`]: Add `OpenDNS` to the default configuration as an allowed identifier
  [#6783](https://github.com/rust-lang/rust-clippy/pull/6783)
* [`missing_panics_doc`]: No longer lints on [`unreachable!`](https://doc.rust-lang.org/std/macro.unreachable.html)
  [#6700](https://github.com/rust-lang/rust-clippy/pull/6700)
* [`collapsible_if`]: No longer lints on if statements with attributes
  [#6701](https://github.com/rust-lang/rust-clippy/pull/6701)
* [`match_same_arms`]: Only considers empty blocks as equal if the tokens contained are the same
  [#6843](https://github.com/rust-lang/rust-clippy/pull/6843)
* [`redundant_closure`]: Now ignores macros
  [#6871](https://github.com/rust-lang/rust-clippy/pull/6871)
* [`manual_map`]: Fixed false positives when control flow statements like `return`, `break` etc. are used
  [#6801](https://github.com/rust-lang/rust-clippy/pull/6801)
* [`vec_init_then_push`]: Fixed false positives for loops and if statements
  [#6697](https://github.com/rust-lang/rust-clippy/pull/6697)
* [`len_without_is_empty`]: Will now consider multiple impl blocks and `#[allow]` on
  the `len` method as well as the type definition.
  [#6853](https://github.com/rust-lang/rust-clippy/pull/6853)
* [`let_underscore_drop`]: Only lints on types which implement `Drop`
  [#6682](https://github.com/rust-lang/rust-clippy/pull/6682)
* [`unit_arg`]: No longer lints on unit arguments when they come from a path expression.
  [#6601](https://github.com/rust-lang/rust-clippy/pull/6601)
* [`cargo_common_metadata`]: No longer lints if
  [`publish = false`](https://doc.rust-lang.org/cargo/reference/manifest.html#the-publish-field)
  is defined in the manifest
  [#6650](https://github.com/rust-lang/rust-clippy/pull/6650)

### Suggestion Fixes/Improvements

* [`collapsible_match`]: Fixed lint message capitalization
  [#6766](https://github.com/rust-lang/rust-clippy/pull/6766)
* [`or_fun_call`]: Improved suggestions for `or_insert(vec![])`
  [#6790](https://github.com/rust-lang/rust-clippy/pull/6790)
* [`manual_map`]: No longer expands macros in the suggestions
  [#6801](https://github.com/rust-lang/rust-clippy/pull/6801)
* Aligned Clippy's lint messages with the rustc dev guide
  [#6787](https://github.com/rust-lang/rust-clippy/pull/6787)

### ICE Fixes

* [`zero_sized_map_values`]
  [#6866](https://github.com/rust-lang/rust-clippy/pull/6866)

### Documentation Improvements

* [`useless_format`]: Improved the documentation example
  [#6854](https://github.com/rust-lang/rust-clippy/pull/6854)
* Clippy's [`README.md`]: Includes a new subsection on running Clippy as a rustc wrapper
  [#6782](https://github.com/rust-lang/rust-clippy/pull/6782)

### Others
* Running `cargo clippy` after `cargo check` now works as expected
  (`cargo clippy` and `cargo check` no longer shares the same build cache)
  [#6687](https://github.com/rust-lang/rust-clippy/pull/6687)
* Cargo now re-runs Clippy if arguments after `--` provided to `cargo clippy` are changed.
  [#6834](https://github.com/rust-lang/rust-clippy/pull/6834)
* Extracted Clippy's `utils` module into the new `clippy_utils` crate
  [#6756](https://github.com/rust-lang/rust-clippy/pull/6756)
* Clippy lintcheck tool improvements
  [#6800](https://github.com/rust-lang/rust-clippy/pull/6800)
  [#6735](https://github.com/rust-lang/rust-clippy/pull/6735)
  [#6764](https://github.com/rust-lang/rust-clippy/pull/6764)
  [#6708](https://github.com/rust-lang/rust-clippy/pull/6708)
  [#6780](https://github.com/rust-lang/rust-clippy/pull/6780)
  [#6686](https://github.com/rust-lang/rust-clippy/pull/6686)

## Rust 1.51

Released 2021-03-25

[View all 78 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2020-12-21T15%3A43%3A04Z..2021-02-03T04%3A21%3A10Z+base%3Amaster)

### New Lints

* [`upper_case_acronyms`]
  [#6475](https://github.com/rust-lang/rust-clippy/pull/6475)
* [`from_over_into`] [#6476](https://github.com/rust-lang/rust-clippy/pull/6476)
* [`case_sensitive_file_extension_comparisons`]
  [#6500](https://github.com/rust-lang/rust-clippy/pull/6500)
* [`needless_question_mark`]
  [#6507](https://github.com/rust-lang/rust-clippy/pull/6507)
* [`missing_panics_doc`]
  [#6523](https://github.com/rust-lang/rust-clippy/pull/6523)
* [`redundant_slicing`]
  [#6528](https://github.com/rust-lang/rust-clippy/pull/6528)
* [`vec_init_then_push`]
  [#6538](https://github.com/rust-lang/rust-clippy/pull/6538)
* [`ptr_as_ptr`] [#6542](https://github.com/rust-lang/rust-clippy/pull/6542)
* [`collapsible_else_if`] (split out from `collapsible_if`)
  [#6544](https://github.com/rust-lang/rust-clippy/pull/6544)
* [`inspect_for_each`] [#6577](https://github.com/rust-lang/rust-clippy/pull/6577)
* [`manual_filter_map`]
  [#6591](https://github.com/rust-lang/rust-clippy/pull/6591)
* [`exhaustive_enums`]
  [#6617](https://github.com/rust-lang/rust-clippy/pull/6617)
* [`exhaustive_structs`]
  [#6617](https://github.com/rust-lang/rust-clippy/pull/6617)

### Moves and Deprecations

* Replace [`find_map`] with [`manual_find_map`]
  [#6591](https://github.com/rust-lang/rust-clippy/pull/6591)
* `unknown_clippy_lints` Now integrated in the `unknown_lints` rustc lint
  [#6653](https://github.com/rust-lang/rust-clippy/pull/6653)

### Enhancements

* [`ptr_arg`] Now also suggests to use `&Path` instead of `&PathBuf`
  [#6506](https://github.com/rust-lang/rust-clippy/pull/6506)
* [`cast_ptr_alignment`] Also lint when the `pointer::cast` method is used
  [#6557](https://github.com/rust-lang/rust-clippy/pull/6557)
* [`collapsible_match`] Now also deals with `&` and `*` operators in the `match`
  scrutinee [#6619](https://github.com/rust-lang/rust-clippy/pull/6619)

### False Positive Fixes

* [`similar_names`] Ignore underscore prefixed names
  [#6403](https://github.com/rust-lang/rust-clippy/pull/6403)
* [`print_literal`] and [`write_literal`] No longer lint numeric literals
  [#6408](https://github.com/rust-lang/rust-clippy/pull/6408)
* [`large_enum_variant`] No longer lints in external macros
  [#6485](https://github.com/rust-lang/rust-clippy/pull/6485)
* [`empty_enum`] Only lint if `never_type` feature is enabled
  [#6513](https://github.com/rust-lang/rust-clippy/pull/6513)
* [`field_reassign_with_default`] No longer lints in macros
  [#6553](https://github.com/rust-lang/rust-clippy/pull/6553)
* [`size_of_in_element_count`] No longer lints when dividing by element size
  [#6578](https://github.com/rust-lang/rust-clippy/pull/6578)
* [`needless_return`] No longer lints in macros
  [#6586](https://github.com/rust-lang/rust-clippy/pull/6586)
* [`match_overlapping_arm`] No longer lint when first arm is completely included
  in second arm [#6603](https://github.com/rust-lang/rust-clippy/pull/6603)
* [`doc_markdown`] Add `WebGL` to the default configuration as an allowed
  identifier [#6605](https://github.com/rust-lang/rust-clippy/pull/6605)

### Suggestion Fixes/Improvements

* [`field_reassign_with_default`] Don't expand macro in lint suggestion
  [#6531](https://github.com/rust-lang/rust-clippy/pull/6531)
* [`match_like_matches_macro`] Strip references in suggestion
  [#6532](https://github.com/rust-lang/rust-clippy/pull/6532)
* [`single_match`] Suggest `if` over `if let` when possible
  [#6574](https://github.com/rust-lang/rust-clippy/pull/6574)
* `ref_in_deref` Use parentheses correctly in suggestion
  [#6609](https://github.com/rust-lang/rust-clippy/pull/6609)
* [`stable_sort_primitive`] Clarify error message
  [#6611](https://github.com/rust-lang/rust-clippy/pull/6611)

### ICE Fixes

* [`zero_sized_map_values`]
  [#6582](https://github.com/rust-lang/rust-clippy/pull/6582)

### Documentation Improvements

* Improve search performance on the Clippy website and make it possible to
  directly search for lints on the GitHub issue tracker
  [#6483](https://github.com/rust-lang/rust-clippy/pull/6483)
* Clean up `README.md` by removing outdated paragraph
  [#6488](https://github.com/rust-lang/rust-clippy/pull/6488)
* [`await_holding_refcell_ref`] and [`await_holding_lock`]
  [#6585](https://github.com/rust-lang/rust-clippy/pull/6585)
* [`as_conversions`] [#6608](https://github.com/rust-lang/rust-clippy/pull/6608)

### Others

* Clippy now has a [Roadmap] for 2021. If you like to get involved in a bigger
  project, take a look at the [Roadmap project page]. All issues listed there
  are actively mentored
  [#6462](https://github.com/rust-lang/rust-clippy/pull/6462)
* The Clippy version number now corresponds to the Rust version number
  [#6526](https://github.com/rust-lang/rust-clippy/pull/6526)
* Fix oversight which caused Clippy to lint deps in some environments, where
  `CLIPPY_TESTS=true` was set somewhere
  [#6575](https://github.com/rust-lang/rust-clippy/pull/6575)
* Add `cargo dev-lintcheck` tool to the Clippy Dev Tool
  [#6469](https://github.com/rust-lang/rust-clippy/pull/6469)

[Roadmap]: https://github.com/rust-lang/rust-clippy/blob/master/book/src/development/proposals/roadmap-2021.md
[Roadmap project page]: https://github.com/rust-lang/rust-clippy/projects/3

## Rust 1.50

Released 2021-02-11

[View all 119 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2020-11-06T18%3A32%3A40Z..2021-01-03T14%3A51%3A18Z+base%3Amaster)

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
* Deprecate `panic_params` lint. This is now available in rustc as `non_fmt_panics`
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
* [`field_reassign_with_default`] No longer lint for private fields
  [#6537](https://github.com/rust-lang/rust-clippy/pull/6537)


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

Released 2020-12-31

[View all 107 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2020-09-24T14%3A05%3A12Z..2020-11-05T13%3A35%3A44Z+base%3Amaster)

### New Lints

* [`field_reassign_with_default`] [#5911](https://github.com/rust-lang/rust-clippy/pull/5911)
* [`await_holding_refcell_ref`] [#6029](https://github.com/rust-lang/rust-clippy/pull/6029)
* [`disallowed_methods`] [#6081](https://github.com/rust-lang/rust-clippy/pull/6081)
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
* Deprecate `drop_bounds` (uplifted)
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

[View all 99 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2020-08-11T13%3A14%3A38Z..2020-09-23T18%3A55%3A22Z+base%3Amaster)

### New lints

* [`self_assignment`] [#5894](https://github.com/rust-lang/rust-clippy/pull/5894)
* [`unnecessary_lazy_evaluations`] [#5720](https://github.com/rust-lang/rust-clippy/pull/5720)
* [`manual_strip`] [#6038](https://github.com/rust-lang/rust-clippy/pull/6038)
* [`map_err_ignore`] [#5998](https://github.com/rust-lang/rust-clippy/pull/5998)
* [`rc_buffer`] [#6044](https://github.com/rust-lang/rust-clippy/pull/6044)
* `to_string_in_display` [#5831](https://github.com/rust-lang/rust-clippy/pull/5831)
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
* `invalid_atomic_ordering`: detect misuse of `compare_exchange`, `compare_exchange_weak`, and `fetch_update`
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
* `to_string_in_display`: avoid linting when calling `to_string()` on anything that is not `self`
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
* [`box_vec`](https://rust-lang.github.io/rust-clippy/master/index.html#box_collection), [`vec_box`] and [`borrowed_box`]: add link to the documentation of `Box`
  [#6023](https://github.com/rust-lang/rust-clippy/pull/6023)

## Rust 1.47

Released 2020-10-08

[View all 76 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2020-06-23T16%3A27%3A11Z..2020-08-11T12%3A52%3A41Z+base%3Amaster)

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

[View all 48 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2020-05-31T12%3A50%3A53Z..2020-06-23T15%3A00%3A32Z+base%3Amaster)

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
* [`disallowed_names`]: Remove `bar` from the default configuration
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

[View all 81 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2020-04-18T20%3A18%3A04Z..2020-05-27T19%3A25%3A04Z+base%3Amaster)

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

[View all 124 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2020-03-05T17%3A30%3A53Z..2020-04-18T09%3A20%3A51Z+base%3Amaster)

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

[View all 91 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2020-01-26T16%3A01%3A11Z..2020-03-04T16%3A45%3A37Z+base%3Amaster)

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

[View all 101 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2019-12-15T01%3A40%3A34Z..2020-01-26T11%3A22%3A13Z+base%3Amaster)

### New lints

* [`filetype_is_file`] [#4543](https://github.com/rust-lang/rust-clippy/pull/4543)
* [`let_underscore_must_use`] [#4823](https://github.com/rust-lang/rust-clippy/pull/4823)
* [`modulo_arithmetic`] [#4867](https://github.com/rust-lang/rust-clippy/pull/4867)
* [`mem_replace_with_default`] [#4881](https://github.com/rust-lang/rust-clippy/pull/4881)
* [`mutable_key_type`] [#4885](https://github.com/rust-lang/rust-clippy/pull/4885)
* [`option_as_ref_deref`] [#4945](https://github.com/rust-lang/rust-clippy/pull/4945)
* [`wildcard_in_or_patterns`] [#4960](https://github.com/rust-lang/rust-clippy/pull/4960)
* [`iter_nth_zero`] [#4966](https://github.com/rust-lang/rust-clippy/pull/4966)
* `invalid_atomic_ordering` [#4999](https://github.com/rust-lang/rust-clippy/pull/4999)
* [`skip_while_next`] [#5067](https://github.com/rust-lang/rust-clippy/pull/5067)

### Moves and Deprecations

* Move [`transmute_float_to_int`] from nursery to complexity group
  [#5015](https://github.com/rust-lang/rust-clippy/pull/5015)
* Move [`range_plus_one`] to pedantic group [#5057](https://github.com/rust-lang/rust-clippy/pull/5057)
* Move [`debug_assert_with_mut_call`] to nursery group [#5106](https://github.com/rust-lang/rust-clippy/pull/5106)
* Deprecate `unused_label` [#4930](https://github.com/rust-lang/rust-clippy/pull/4930)

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
* `unknown_clippy_lints` [#4963](https://github.com/rust-lang/rust-clippy/pull/4963)
* [`explicit_into_iter_loop`] [#4978](https://github.com/rust-lang/rust-clippy/pull/4978)
* [`useless_attribute`] [#5022](https://github.com/rust-lang/rust-clippy/pull/5022)
* `if_let_some_result` [#5032](https://github.com/rust-lang/rust-clippy/pull/5032)

### ICE fixes

* [`unsound_collection_transmute`] [#4975](https://github.com/rust-lang/rust-clippy/pull/4975)

### Documentation

* Improve documentation of [`empty_enum`], [`replace_consts`], [`redundant_clone`], and [`iterator_step_by_zero`]


## Rust 1.41

Released 2020-01-30

[View all 74 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2019-10-28T20%3A50%3A24Z..2019-12-12T00%3A53%3A03Z+base%3Amaster)

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
* Deprecate `into_iter_on_array` [#4788](https://github.com/rust-lang/rust-clippy/pull/4788)
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

[View all 76 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2019-09-23T06%3A18%3A04Z..2019-10-28T17%3A34%3A55Z+base%3Amaster)

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

[View all 100 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2019-08-11T19%3A21%3A38Z..2019-09-22T12%3A07%3A39Z+base%3Amaster)

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

[View all 76 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2019-06-30T13%3A40%3A26Z..2019-08-11T09%3A47%3A27Z+base%3Amaster)

* New Lints:
  * [`main_recursion`] [#4203](https://github.com/rust-lang/rust-clippy/pull/4203)
  * [`inherent_to_string`] [#4259](https://github.com/rust-lang/rust-clippy/pull/4259)
  * [`inherent_to_string_shadow_display`] [#4259](https://github.com/rust-lang/rust-clippy/pull/4259)
  * [`type_repetition_in_bounds`] [#3766](https://github.com/rust-lang/rust-clippy/pull/3766)
  * [`try_err`] [#4222](https://github.com/rust-lang/rust-clippy/pull/4222)
* Move `{unnecessary,panicking}_unwrap` out of nursery [#4307](https://github.com/rust-lang/rust-clippy/pull/4307)
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

[View all 72 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2019-05-19T08%3A11%3A23Z..2019-06-25T23%3A22%3A22Z+base%3Amaster)

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

[View all 81 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2019-04-10T09%3A41%3A56Z..2019-05-18T00%3A29%3A40Z+base%3Amaster)

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

* New lint: `drop_bounds` to detect `T: Drop` bounds
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

[View all 61 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2019-01-17T17%3A45%3A39Z..2019-02-19T08%3A24%3A05Z+base%3Amaster)

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

[View all 120 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2018-11-28T06%3A19%3A50Z..2019-01-15T09%3A27%3A02Z+base%3Amaster)

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

[View all 71 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2018-10-24T05%3A02%3A21Z..2018-11-27T17%3A29%3A34Z+base%3Amaster)

* New lints: [`slow_vector_initialization`], `mem_discriminant_non_enum`,
  [`redundant_clone`], [`wildcard_dependencies`],
  [`into_iter_on_ref`], `into_iter_on_array`, [`deprecated_cfg_attr`],
  [`cargo_common_metadata`]
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

[View all 88 merged pull requests](https://github.com/rust-lang/rust-clippy/pulls?q=merged%3A2018-08-02T16%3A54%3A12Z..2018-09-17T09%3A44%3A06Z+base%3Amaster)
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
* New lints: [`explicit_write`], `identity_conversion`, [`implicit_hasher`], `invalid_ref`, [`option_map_or_none`],
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
* New lints: `temporary_cstring_as_ptr`, [`unsafe_removed_from_name`], and [`mem_forget`]

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
* New lints: [`overflow_check_conditional`], `unused_label`, [`new_without_default`]

## 0.0.48 — 2016-03-07
* Fixed: ICE in [`needless_range_loop`] with globals

## 0.0.47 — 2016-03-07
* Update to *rustc 1.9.0-nightly (998a6720b 2016-03-07)*
* New lint: [`redundant_closure_call`]

[`AsMut`]: https://doc.rust-lang.org/std/convert/trait.AsMut.html
[`AsRef`]: https://doc.rust-lang.org/std/convert/trait.AsRef.html
[configuration file]: ./rust-clippy#configuration
[pull3665]: https://github.com/rust-lang/rust-clippy/pull/3665
[adding_lints]: https://github.com/rust-lang/rust-clippy/blob/master/book/src/development/adding_lints.md
[`README.md`]: https://github.com/rust-lang/rust-clippy/blob/master/README.md
[Clippy's lint list]: https://rust-lang.github.io/rust-clippy/master/index.html

<!-- lint disable no-unused-definitions -->
<!-- begin autogenerated links to lint list -->
[`absolute_paths`]: https://rust-lang.github.io/rust-clippy/master/index.html#absolute_paths
[`absurd_extreme_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#absurd_extreme_comparisons
[`alloc_instead_of_core`]: https://rust-lang.github.io/rust-clippy/master/index.html#alloc_instead_of_core
[`allow_attributes`]: https://rust-lang.github.io/rust-clippy/master/index.html#allow_attributes
[`allow_attributes_without_reason`]: https://rust-lang.github.io/rust-clippy/master/index.html#allow_attributes_without_reason
[`almost_complete_letter_range`]: https://rust-lang.github.io/rust-clippy/master/index.html#almost_complete_letter_range
[`almost_complete_range`]: https://rust-lang.github.io/rust-clippy/master/index.html#almost_complete_range
[`almost_swapped`]: https://rust-lang.github.io/rust-clippy/master/index.html#almost_swapped
[`approx_constant`]: https://rust-lang.github.io/rust-clippy/master/index.html#approx_constant
[`arbitrary_source_item_ordering`]: https://rust-lang.github.io/rust-clippy/master/index.html#arbitrary_source_item_ordering
[`arc_with_non_send_sync`]: https://rust-lang.github.io/rust-clippy/master/index.html#arc_with_non_send_sync
[`arithmetic_side_effects`]: https://rust-lang.github.io/rust-clippy/master/index.html#arithmetic_side_effects
[`as_conversions`]: https://rust-lang.github.io/rust-clippy/master/index.html#as_conversions
[`as_pointer_underscore`]: https://rust-lang.github.io/rust-clippy/master/index.html#as_pointer_underscore
[`as_ptr_cast_mut`]: https://rust-lang.github.io/rust-clippy/master/index.html#as_ptr_cast_mut
[`as_underscore`]: https://rust-lang.github.io/rust-clippy/master/index.html#as_underscore
[`assertions_on_constants`]: https://rust-lang.github.io/rust-clippy/master/index.html#assertions_on_constants
[`assertions_on_result_states`]: https://rust-lang.github.io/rust-clippy/master/index.html#assertions_on_result_states
[`assign_op_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#assign_op_pattern
[`assign_ops`]: https://rust-lang.github.io/rust-clippy/master/index.html#assign_ops
[`assigning_clones`]: https://rust-lang.github.io/rust-clippy/master/index.html#assigning_clones
[`async_yields_async`]: https://rust-lang.github.io/rust-clippy/master/index.html#async_yields_async
[`await_holding_invalid_type`]: https://rust-lang.github.io/rust-clippy/master/index.html#await_holding_invalid_type
[`await_holding_lock`]: https://rust-lang.github.io/rust-clippy/master/index.html#await_holding_lock
[`await_holding_refcell_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#await_holding_refcell_ref
[`bad_bit_mask`]: https://rust-lang.github.io/rust-clippy/master/index.html#bad_bit_mask
[`big_endian_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#big_endian_bytes
[`bind_instead_of_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#bind_instead_of_map
[`blacklisted_name`]: https://rust-lang.github.io/rust-clippy/master/index.html#blacklisted_name
[`blanket_clippy_restriction_lints`]: https://rust-lang.github.io/rust-clippy/master/index.html#blanket_clippy_restriction_lints
[`block_in_if_condition_expr`]: https://rust-lang.github.io/rust-clippy/master/index.html#block_in_if_condition_expr
[`block_in_if_condition_stmt`]: https://rust-lang.github.io/rust-clippy/master/index.html#block_in_if_condition_stmt
[`blocks_in_conditions`]: https://rust-lang.github.io/rust-clippy/master/index.html#blocks_in_conditions
[`blocks_in_if_conditions`]: https://rust-lang.github.io/rust-clippy/master/index.html#blocks_in_if_conditions
[`bool_assert_comparison`]: https://rust-lang.github.io/rust-clippy/master/index.html#bool_assert_comparison
[`bool_comparison`]: https://rust-lang.github.io/rust-clippy/master/index.html#bool_comparison
[`bool_to_int_with_if`]: https://rust-lang.github.io/rust-clippy/master/index.html#bool_to_int_with_if
[`borrow_as_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#borrow_as_ptr
[`borrow_deref_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#borrow_deref_ref
[`borrow_interior_mutable_const`]: https://rust-lang.github.io/rust-clippy/master/index.html#borrow_interior_mutable_const
[`borrowed_box`]: https://rust-lang.github.io/rust-clippy/master/index.html#borrowed_box
[`box_collection`]: https://rust-lang.github.io/rust-clippy/master/index.html#box_collection
[`box_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#box_default
[`box_vec`]: https://rust-lang.github.io/rust-clippy/master/index.html#box_vec
[`boxed_local`]: https://rust-lang.github.io/rust-clippy/master/index.html#boxed_local
[`branches_sharing_code`]: https://rust-lang.github.io/rust-clippy/master/index.html#branches_sharing_code
[`builtin_type_shadow`]: https://rust-lang.github.io/rust-clippy/master/index.html#builtin_type_shadow
[`byte_char_slices`]: https://rust-lang.github.io/rust-clippy/master/index.html#byte_char_slices
[`bytes_count_to_len`]: https://rust-lang.github.io/rust-clippy/master/index.html#bytes_count_to_len
[`bytes_nth`]: https://rust-lang.github.io/rust-clippy/master/index.html#bytes_nth
[`cargo_common_metadata`]: https://rust-lang.github.io/rust-clippy/master/index.html#cargo_common_metadata
[`case_sensitive_file_extension_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#case_sensitive_file_extension_comparisons
[`cast_abs_to_unsigned`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_abs_to_unsigned
[`cast_enum_constructor`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_enum_constructor
[`cast_enum_truncation`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_enum_truncation
[`cast_lossless`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_lossless
[`cast_nan_to_int`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_nan_to_int
[`cast_possible_truncation`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_possible_truncation
[`cast_possible_wrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_possible_wrap
[`cast_precision_loss`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_precision_loss
[`cast_ptr_alignment`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_ptr_alignment
[`cast_ref_to_mut`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_ref_to_mut
[`cast_sign_loss`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_sign_loss
[`cast_slice_different_sizes`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_slice_different_sizes
[`cast_slice_from_raw_parts`]: https://rust-lang.github.io/rust-clippy/master/index.html#cast_slice_from_raw_parts
[`cfg_not_test`]: https://rust-lang.github.io/rust-clippy/master/index.html#cfg_not_test
[`char_indices_as_byte_indices`]: https://rust-lang.github.io/rust-clippy/master/index.html#char_indices_as_byte_indices
[`char_lit_as_u8`]: https://rust-lang.github.io/rust-clippy/master/index.html#char_lit_as_u8
[`chars_last_cmp`]: https://rust-lang.github.io/rust-clippy/master/index.html#chars_last_cmp
[`chars_next_cmp`]: https://rust-lang.github.io/rust-clippy/master/index.html#chars_next_cmp
[`checked_conversions`]: https://rust-lang.github.io/rust-clippy/master/index.html#checked_conversions
[`clear_with_drain`]: https://rust-lang.github.io/rust-clippy/master/index.html#clear_with_drain
[`clone_double_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#clone_double_ref
[`clone_on_copy`]: https://rust-lang.github.io/rust-clippy/master/index.html#clone_on_copy
[`clone_on_ref_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#clone_on_ref_ptr
[`cloned_instead_of_copied`]: https://rust-lang.github.io/rust-clippy/master/index.html#cloned_instead_of_copied
[`cloned_ref_to_slice_refs`]: https://rust-lang.github.io/rust-clippy/master/index.html#cloned_ref_to_slice_refs
[`cmp_nan`]: https://rust-lang.github.io/rust-clippy/master/index.html#cmp_nan
[`cmp_null`]: https://rust-lang.github.io/rust-clippy/master/index.html#cmp_null
[`cmp_owned`]: https://rust-lang.github.io/rust-clippy/master/index.html#cmp_owned
[`cognitive_complexity`]: https://rust-lang.github.io/rust-clippy/master/index.html#cognitive_complexity
[`collapsible_else_if`]: https://rust-lang.github.io/rust-clippy/master/index.html#collapsible_else_if
[`collapsible_if`]: https://rust-lang.github.io/rust-clippy/master/index.html#collapsible_if
[`collapsible_match`]: https://rust-lang.github.io/rust-clippy/master/index.html#collapsible_match
[`collapsible_str_replace`]: https://rust-lang.github.io/rust-clippy/master/index.html#collapsible_str_replace
[`collection_is_never_read`]: https://rust-lang.github.io/rust-clippy/master/index.html#collection_is_never_read
[`comparison_chain`]: https://rust-lang.github.io/rust-clippy/master/index.html#comparison_chain
[`comparison_to_empty`]: https://rust-lang.github.io/rust-clippy/master/index.html#comparison_to_empty
[`confusing_method_to_numeric_cast`]: https://rust-lang.github.io/rust-clippy/master/index.html#confusing_method_to_numeric_cast
[`const_is_empty`]: https://rust-lang.github.io/rust-clippy/master/index.html#const_is_empty
[`const_static_lifetime`]: https://rust-lang.github.io/rust-clippy/master/index.html#const_static_lifetime
[`copy_iterator`]: https://rust-lang.github.io/rust-clippy/master/index.html#copy_iterator
[`crate_in_macro_def`]: https://rust-lang.github.io/rust-clippy/master/index.html#crate_in_macro_def
[`create_dir`]: https://rust-lang.github.io/rust-clippy/master/index.html#create_dir
[`crosspointer_transmute`]: https://rust-lang.github.io/rust-clippy/master/index.html#crosspointer_transmute
[`cyclomatic_complexity`]: https://rust-lang.github.io/rust-clippy/master/index.html#cyclomatic_complexity
[`dbg_macro`]: https://rust-lang.github.io/rust-clippy/master/index.html#dbg_macro
[`debug_assert_with_mut_call`]: https://rust-lang.github.io/rust-clippy/master/index.html#debug_assert_with_mut_call
[`decimal_literal_representation`]: https://rust-lang.github.io/rust-clippy/master/index.html#decimal_literal_representation
[`declare_interior_mutable_const`]: https://rust-lang.github.io/rust-clippy/master/index.html#declare_interior_mutable_const
[`default_constructed_unit_structs`]: https://rust-lang.github.io/rust-clippy/master/index.html#default_constructed_unit_structs
[`default_instead_of_iter_empty`]: https://rust-lang.github.io/rust-clippy/master/index.html#default_instead_of_iter_empty
[`default_numeric_fallback`]: https://rust-lang.github.io/rust-clippy/master/index.html#default_numeric_fallback
[`default_trait_access`]: https://rust-lang.github.io/rust-clippy/master/index.html#default_trait_access
[`default_union_representation`]: https://rust-lang.github.io/rust-clippy/master/index.html#default_union_representation
[`deprecated_cfg_attr`]: https://rust-lang.github.io/rust-clippy/master/index.html#deprecated_cfg_attr
[`deprecated_clippy_cfg_attr`]: https://rust-lang.github.io/rust-clippy/master/index.html#deprecated_clippy_cfg_attr
[`deprecated_semver`]: https://rust-lang.github.io/rust-clippy/master/index.html#deprecated_semver
[`deref_addrof`]: https://rust-lang.github.io/rust-clippy/master/index.html#deref_addrof
[`deref_by_slicing`]: https://rust-lang.github.io/rust-clippy/master/index.html#deref_by_slicing
[`derivable_impls`]: https://rust-lang.github.io/rust-clippy/master/index.html#derivable_impls
[`derive_hash_xor_eq`]: https://rust-lang.github.io/rust-clippy/master/index.html#derive_hash_xor_eq
[`derive_ord_xor_partial_ord`]: https://rust-lang.github.io/rust-clippy/master/index.html#derive_ord_xor_partial_ord
[`derive_partial_eq_without_eq`]: https://rust-lang.github.io/rust-clippy/master/index.html#derive_partial_eq_without_eq
[`derived_hash_with_manual_eq`]: https://rust-lang.github.io/rust-clippy/master/index.html#derived_hash_with_manual_eq
[`disallowed_macros`]: https://rust-lang.github.io/rust-clippy/master/index.html#disallowed_macros
[`disallowed_method`]: https://rust-lang.github.io/rust-clippy/master/index.html#disallowed_method
[`disallowed_methods`]: https://rust-lang.github.io/rust-clippy/master/index.html#disallowed_methods
[`disallowed_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#disallowed_names
[`disallowed_script_idents`]: https://rust-lang.github.io/rust-clippy/master/index.html#disallowed_script_idents
[`disallowed_type`]: https://rust-lang.github.io/rust-clippy/master/index.html#disallowed_type
[`disallowed_types`]: https://rust-lang.github.io/rust-clippy/master/index.html#disallowed_types
[`diverging_sub_expression`]: https://rust-lang.github.io/rust-clippy/master/index.html#diverging_sub_expression
[`doc_comment_double_space_linebreaks`]: https://rust-lang.github.io/rust-clippy/master/index.html#doc_comment_double_space_linebreaks
[`doc_include_without_cfg`]: https://rust-lang.github.io/rust-clippy/master/index.html#doc_include_without_cfg
[`doc_lazy_continuation`]: https://rust-lang.github.io/rust-clippy/master/index.html#doc_lazy_continuation
[`doc_link_code`]: https://rust-lang.github.io/rust-clippy/master/index.html#doc_link_code
[`doc_link_with_quotes`]: https://rust-lang.github.io/rust-clippy/master/index.html#doc_link_with_quotes
[`doc_markdown`]: https://rust-lang.github.io/rust-clippy/master/index.html#doc_markdown
[`doc_nested_refdefs`]: https://rust-lang.github.io/rust-clippy/master/index.html#doc_nested_refdefs
[`doc_overindented_list_items`]: https://rust-lang.github.io/rust-clippy/master/index.html#doc_overindented_list_items
[`double_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#double_comparisons
[`double_ended_iterator_last`]: https://rust-lang.github.io/rust-clippy/master/index.html#double_ended_iterator_last
[`double_must_use`]: https://rust-lang.github.io/rust-clippy/master/index.html#double_must_use
[`double_neg`]: https://rust-lang.github.io/rust-clippy/master/index.html#double_neg
[`double_parens`]: https://rust-lang.github.io/rust-clippy/master/index.html#double_parens
[`drain_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#drain_collect
[`drop_bounds`]: https://rust-lang.github.io/rust-clippy/master/index.html#drop_bounds
[`drop_copy`]: https://rust-lang.github.io/rust-clippy/master/index.html#drop_copy
[`drop_non_drop`]: https://rust-lang.github.io/rust-clippy/master/index.html#drop_non_drop
[`drop_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#drop_ref
[`duplicate_mod`]: https://rust-lang.github.io/rust-clippy/master/index.html#duplicate_mod
[`duplicate_underscore_argument`]: https://rust-lang.github.io/rust-clippy/master/index.html#duplicate_underscore_argument
[`duplicated_attributes`]: https://rust-lang.github.io/rust-clippy/master/index.html#duplicated_attributes
[`duration_subsec`]: https://rust-lang.github.io/rust-clippy/master/index.html#duration_subsec
[`eager_transmute`]: https://rust-lang.github.io/rust-clippy/master/index.html#eager_transmute
[`elidable_lifetime_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#elidable_lifetime_names
[`else_if_without_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#else_if_without_else
[`empty_docs`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_docs
[`empty_drop`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_drop
[`empty_enum`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_enum
[`empty_enum_variants_with_brackets`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_enum_variants_with_brackets
[`empty_line_after_doc_comments`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_line_after_doc_comments
[`empty_line_after_outer_attr`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_line_after_outer_attr
[`empty_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_loop
[`empty_structs_with_brackets`]: https://rust-lang.github.io/rust-clippy/master/index.html#empty_structs_with_brackets
[`enum_clike_unportable_variant`]: https://rust-lang.github.io/rust-clippy/master/index.html#enum_clike_unportable_variant
[`enum_glob_use`]: https://rust-lang.github.io/rust-clippy/master/index.html#enum_glob_use
[`enum_variant_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#enum_variant_names
[`eq_op`]: https://rust-lang.github.io/rust-clippy/master/index.html#eq_op
[`equatable_if_let`]: https://rust-lang.github.io/rust-clippy/master/index.html#equatable_if_let
[`erasing_op`]: https://rust-lang.github.io/rust-clippy/master/index.html#erasing_op
[`err_expect`]: https://rust-lang.github.io/rust-clippy/master/index.html#err_expect
[`error_impl_error`]: https://rust-lang.github.io/rust-clippy/master/index.html#error_impl_error
[`eval_order_dependence`]: https://rust-lang.github.io/rust-clippy/master/index.html#eval_order_dependence
[`excessive_nesting`]: https://rust-lang.github.io/rust-clippy/master/index.html#excessive_nesting
[`excessive_precision`]: https://rust-lang.github.io/rust-clippy/master/index.html#excessive_precision
[`exhaustive_enums`]: https://rust-lang.github.io/rust-clippy/master/index.html#exhaustive_enums
[`exhaustive_structs`]: https://rust-lang.github.io/rust-clippy/master/index.html#exhaustive_structs
[`exit`]: https://rust-lang.github.io/rust-clippy/master/index.html#exit
[`expect_fun_call`]: https://rust-lang.github.io/rust-clippy/master/index.html#expect_fun_call
[`expect_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#expect_used
[`expl_impl_clone_on_copy`]: https://rust-lang.github.io/rust-clippy/master/index.html#expl_impl_clone_on_copy
[`explicit_auto_deref`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_auto_deref
[`explicit_counter_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_counter_loop
[`explicit_deref_methods`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_deref_methods
[`explicit_into_iter_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_into_iter_loop
[`explicit_iter_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_iter_loop
[`explicit_write`]: https://rust-lang.github.io/rust-clippy/master/index.html#explicit_write
[`extend_from_slice`]: https://rust-lang.github.io/rust-clippy/master/index.html#extend_from_slice
[`extend_with_drain`]: https://rust-lang.github.io/rust-clippy/master/index.html#extend_with_drain
[`extra_unused_lifetimes`]: https://rust-lang.github.io/rust-clippy/master/index.html#extra_unused_lifetimes
[`extra_unused_type_parameters`]: https://rust-lang.github.io/rust-clippy/master/index.html#extra_unused_type_parameters
[`fallible_impl_from`]: https://rust-lang.github.io/rust-clippy/master/index.html#fallible_impl_from
[`field_reassign_with_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#field_reassign_with_default
[`field_scoped_visibility_modifiers`]: https://rust-lang.github.io/rust-clippy/master/index.html#field_scoped_visibility_modifiers
[`filetype_is_file`]: https://rust-lang.github.io/rust-clippy/master/index.html#filetype_is_file
[`filter_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#filter_map
[`filter_map_bool_then`]: https://rust-lang.github.io/rust-clippy/master/index.html#filter_map_bool_then
[`filter_map_identity`]: https://rust-lang.github.io/rust-clippy/master/index.html#filter_map_identity
[`filter_map_next`]: https://rust-lang.github.io/rust-clippy/master/index.html#filter_map_next
[`filter_next`]: https://rust-lang.github.io/rust-clippy/master/index.html#filter_next
[`find_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#find_map
[`flat_map_identity`]: https://rust-lang.github.io/rust-clippy/master/index.html#flat_map_identity
[`flat_map_option`]: https://rust-lang.github.io/rust-clippy/master/index.html#flat_map_option
[`float_arithmetic`]: https://rust-lang.github.io/rust-clippy/master/index.html#float_arithmetic
[`float_cmp`]: https://rust-lang.github.io/rust-clippy/master/index.html#float_cmp
[`float_cmp_const`]: https://rust-lang.github.io/rust-clippy/master/index.html#float_cmp_const
[`float_equality_without_abs`]: https://rust-lang.github.io/rust-clippy/master/index.html#float_equality_without_abs
[`fn_address_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_address_comparisons
[`fn_null_check`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_null_check
[`fn_params_excessive_bools`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_params_excessive_bools
[`fn_to_numeric_cast`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_to_numeric_cast
[`fn_to_numeric_cast_any`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_to_numeric_cast_any
[`fn_to_numeric_cast_with_truncation`]: https://rust-lang.github.io/rust-clippy/master/index.html#fn_to_numeric_cast_with_truncation
[`for_kv_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#for_kv_map
[`for_loop_over_option`]: https://rust-lang.github.io/rust-clippy/master/index.html#for_loop_over_option
[`for_loop_over_result`]: https://rust-lang.github.io/rust-clippy/master/index.html#for_loop_over_result
[`for_loops_over_fallibles`]: https://rust-lang.github.io/rust-clippy/master/index.html#for_loops_over_fallibles
[`forget_copy`]: https://rust-lang.github.io/rust-clippy/master/index.html#forget_copy
[`forget_non_drop`]: https://rust-lang.github.io/rust-clippy/master/index.html#forget_non_drop
[`forget_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#forget_ref
[`format_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#format_collect
[`format_in_format_args`]: https://rust-lang.github.io/rust-clippy/master/index.html#format_in_format_args
[`format_push_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#format_push_string
[`four_forward_slashes`]: https://rust-lang.github.io/rust-clippy/master/index.html#four_forward_slashes
[`from_iter_instead_of_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#from_iter_instead_of_collect
[`from_over_into`]: https://rust-lang.github.io/rust-clippy/master/index.html#from_over_into
[`from_raw_with_void_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#from_raw_with_void_ptr
[`from_str_radix_10`]: https://rust-lang.github.io/rust-clippy/master/index.html#from_str_radix_10
[`future_not_send`]: https://rust-lang.github.io/rust-clippy/master/index.html#future_not_send
[`get_first`]: https://rust-lang.github.io/rust-clippy/master/index.html#get_first
[`get_last_with_len`]: https://rust-lang.github.io/rust-clippy/master/index.html#get_last_with_len
[`get_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#get_unwrap
[`host_endian_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#host_endian_bytes
[`identity_conversion`]: https://rust-lang.github.io/rust-clippy/master/index.html#identity_conversion
[`identity_op`]: https://rust-lang.github.io/rust-clippy/master/index.html#identity_op
[`if_let_mutex`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_let_mutex
[`if_let_redundant_pattern_matching`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_let_redundant_pattern_matching
[`if_let_some_result`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_let_some_result
[`if_not_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_not_else
[`if_same_then_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_same_then_else
[`if_then_some_else_none`]: https://rust-lang.github.io/rust-clippy/master/index.html#if_then_some_else_none
[`ifs_same_cond`]: https://rust-lang.github.io/rust-clippy/master/index.html#ifs_same_cond
[`ignore_without_reason`]: https://rust-lang.github.io/rust-clippy/master/index.html#ignore_without_reason
[`ignored_unit_patterns`]: https://rust-lang.github.io/rust-clippy/master/index.html#ignored_unit_patterns
[`impl_hash_borrow_with_str_and_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#impl_hash_borrow_with_str_and_bytes
[`impl_trait_in_params`]: https://rust-lang.github.io/rust-clippy/master/index.html#impl_trait_in_params
[`implicit_clone`]: https://rust-lang.github.io/rust-clippy/master/index.html#implicit_clone
[`implicit_hasher`]: https://rust-lang.github.io/rust-clippy/master/index.html#implicit_hasher
[`implicit_return`]: https://rust-lang.github.io/rust-clippy/master/index.html#implicit_return
[`implicit_saturating_add`]: https://rust-lang.github.io/rust-clippy/master/index.html#implicit_saturating_add
[`implicit_saturating_sub`]: https://rust-lang.github.io/rust-clippy/master/index.html#implicit_saturating_sub
[`implied_bounds_in_impls`]: https://rust-lang.github.io/rust-clippy/master/index.html#implied_bounds_in_impls
[`impossible_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#impossible_comparisons
[`imprecise_flops`]: https://rust-lang.github.io/rust-clippy/master/index.html#imprecise_flops
[`incompatible_msrv`]: https://rust-lang.github.io/rust-clippy/master/index.html#incompatible_msrv
[`inconsistent_digit_grouping`]: https://rust-lang.github.io/rust-clippy/master/index.html#inconsistent_digit_grouping
[`inconsistent_struct_constructor`]: https://rust-lang.github.io/rust-clippy/master/index.html#inconsistent_struct_constructor
[`incorrect_clone_impl_on_copy_type`]: https://rust-lang.github.io/rust-clippy/master/index.html#incorrect_clone_impl_on_copy_type
[`incorrect_partial_ord_impl_on_ord_type`]: https://rust-lang.github.io/rust-clippy/master/index.html#incorrect_partial_ord_impl_on_ord_type
[`index_refutable_slice`]: https://rust-lang.github.io/rust-clippy/master/index.html#index_refutable_slice
[`indexing_slicing`]: https://rust-lang.github.io/rust-clippy/master/index.html#indexing_slicing
[`ineffective_bit_mask`]: https://rust-lang.github.io/rust-clippy/master/index.html#ineffective_bit_mask
[`ineffective_open_options`]: https://rust-lang.github.io/rust-clippy/master/index.html#ineffective_open_options
[`inefficient_to_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#inefficient_to_string
[`infallible_destructuring_match`]: https://rust-lang.github.io/rust-clippy/master/index.html#infallible_destructuring_match
[`infinite_iter`]: https://rust-lang.github.io/rust-clippy/master/index.html#infinite_iter
[`infinite_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#infinite_loop
[`inherent_to_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#inherent_to_string
[`inherent_to_string_shadow_display`]: https://rust-lang.github.io/rust-clippy/master/index.html#inherent_to_string_shadow_display
[`init_numbered_fields`]: https://rust-lang.github.io/rust-clippy/master/index.html#init_numbered_fields
[`inline_always`]: https://rust-lang.github.io/rust-clippy/master/index.html#inline_always
[`inline_asm_x86_att_syntax`]: https://rust-lang.github.io/rust-clippy/master/index.html#inline_asm_x86_att_syntax
[`inline_asm_x86_intel_syntax`]: https://rust-lang.github.io/rust-clippy/master/index.html#inline_asm_x86_intel_syntax
[`inline_fn_without_body`]: https://rust-lang.github.io/rust-clippy/master/index.html#inline_fn_without_body
[`inspect_for_each`]: https://rust-lang.github.io/rust-clippy/master/index.html#inspect_for_each
[`int_plus_one`]: https://rust-lang.github.io/rust-clippy/master/index.html#int_plus_one
[`integer_arithmetic`]: https://rust-lang.github.io/rust-clippy/master/index.html#integer_arithmetic
[`integer_division`]: https://rust-lang.github.io/rust-clippy/master/index.html#integer_division
[`integer_division_remainder_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#integer_division_remainder_used
[`into_iter_on_array`]: https://rust-lang.github.io/rust-clippy/master/index.html#into_iter_on_array
[`into_iter_on_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#into_iter_on_ref
[`into_iter_without_iter`]: https://rust-lang.github.io/rust-clippy/master/index.html#into_iter_without_iter
[`invalid_atomic_ordering`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_atomic_ordering
[`invalid_null_ptr_usage`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_null_ptr_usage
[`invalid_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_ref
[`invalid_regex`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_regex
[`invalid_upcast_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_upcast_comparisons
[`invalid_utf8_in_unchecked`]: https://rust-lang.github.io/rust-clippy/master/index.html#invalid_utf8_in_unchecked
[`inverted_saturating_sub`]: https://rust-lang.github.io/rust-clippy/master/index.html#inverted_saturating_sub
[`invisible_characters`]: https://rust-lang.github.io/rust-clippy/master/index.html#invisible_characters
[`io_other_error`]: https://rust-lang.github.io/rust-clippy/master/index.html#io_other_error
[`is_digit_ascii_radix`]: https://rust-lang.github.io/rust-clippy/master/index.html#is_digit_ascii_radix
[`items_after_statements`]: https://rust-lang.github.io/rust-clippy/master/index.html#items_after_statements
[`items_after_test_module`]: https://rust-lang.github.io/rust-clippy/master/index.html#items_after_test_module
[`iter_cloned_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_cloned_collect
[`iter_count`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_count
[`iter_filter_is_ok`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_filter_is_ok
[`iter_filter_is_some`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_filter_is_some
[`iter_kv_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_kv_map
[`iter_next_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_next_loop
[`iter_next_slice`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_next_slice
[`iter_not_returning_iterator`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_not_returning_iterator
[`iter_nth`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_nth
[`iter_nth_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_nth_zero
[`iter_on_empty_collections`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_on_empty_collections
[`iter_on_single_items`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_on_single_items
[`iter_out_of_bounds`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_out_of_bounds
[`iter_over_hash_type`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_over_hash_type
[`iter_overeager_cloned`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_overeager_cloned
[`iter_skip_next`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_skip_next
[`iter_skip_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_skip_zero
[`iter_with_drain`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_with_drain
[`iter_without_into_iter`]: https://rust-lang.github.io/rust-clippy/master/index.html#iter_without_into_iter
[`iterator_step_by_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#iterator_step_by_zero
[`join_absolute_paths`]: https://rust-lang.github.io/rust-clippy/master/index.html#join_absolute_paths
[`just_underscores_and_digits`]: https://rust-lang.github.io/rust-clippy/master/index.html#just_underscores_and_digits
[`large_const_arrays`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_const_arrays
[`large_digit_groups`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_digit_groups
[`large_enum_variant`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_enum_variant
[`large_futures`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_futures
[`large_include_file`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_include_file
[`large_stack_arrays`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_stack_arrays
[`large_stack_frames`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_stack_frames
[`large_types_passed_by_value`]: https://rust-lang.github.io/rust-clippy/master/index.html#large_types_passed_by_value
[`legacy_numeric_constants`]: https://rust-lang.github.io/rust-clippy/master/index.html#legacy_numeric_constants
[`len_without_is_empty`]: https://rust-lang.github.io/rust-clippy/master/index.html#len_without_is_empty
[`len_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#len_zero
[`let_and_return`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_and_return
[`let_underscore_drop`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_underscore_drop
[`let_underscore_future`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_underscore_future
[`let_underscore_lock`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_underscore_lock
[`let_underscore_must_use`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_underscore_must_use
[`let_underscore_untyped`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_underscore_untyped
[`let_unit_value`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_unit_value
[`let_with_type_underscore`]: https://rust-lang.github.io/rust-clippy/master/index.html#let_with_type_underscore
[`lines_filter_map_ok`]: https://rust-lang.github.io/rust-clippy/master/index.html#lines_filter_map_ok
[`linkedlist`]: https://rust-lang.github.io/rust-clippy/master/index.html#linkedlist
[`lint_groups_priority`]: https://rust-lang.github.io/rust-clippy/master/index.html#lint_groups_priority
[`literal_string_with_formatting_args`]: https://rust-lang.github.io/rust-clippy/master/index.html#literal_string_with_formatting_args
[`little_endian_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#little_endian_bytes
[`logic_bug`]: https://rust-lang.github.io/rust-clippy/master/index.html#logic_bug
[`lossy_float_literal`]: https://rust-lang.github.io/rust-clippy/master/index.html#lossy_float_literal
[`macro_metavars_in_unsafe`]: https://rust-lang.github.io/rust-clippy/master/index.html#macro_metavars_in_unsafe
[`macro_use_imports`]: https://rust-lang.github.io/rust-clippy/master/index.html#macro_use_imports
[`main_recursion`]: https://rust-lang.github.io/rust-clippy/master/index.html#main_recursion
[`manual_abs_diff`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_abs_diff
[`manual_assert`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_assert
[`manual_async_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_async_fn
[`manual_bits`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_bits
[`manual_c_str_literals`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_c_str_literals
[`manual_clamp`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_clamp
[`manual_contains`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_contains
[`manual_dangling_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_dangling_ptr
[`manual_div_ceil`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_div_ceil
[`manual_filter`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_filter
[`manual_filter_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_filter_map
[`manual_find`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_find
[`manual_find_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_find_map
[`manual_flatten`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_flatten
[`manual_hash_one`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_hash_one
[`manual_ignore_case_cmp`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_ignore_case_cmp
[`manual_inspect`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_inspect
[`manual_instant_elapsed`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_instant_elapsed
[`manual_is_ascii_check`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_is_ascii_check
[`manual_is_finite`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_is_finite
[`manual_is_infinite`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_is_infinite
[`manual_is_power_of_two`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_is_power_of_two
[`manual_is_variant_and`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_is_variant_and
[`manual_let_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_let_else
[`manual_main_separator_str`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_main_separator_str
[`manual_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_map
[`manual_memcpy`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_memcpy
[`manual_midpoint`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_midpoint
[`manual_next_back`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_next_back
[`manual_non_exhaustive`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_non_exhaustive
[`manual_ok_err`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_ok_err
[`manual_ok_or`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_ok_or
[`manual_option_as_slice`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_option_as_slice
[`manual_pattern_char_comparison`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_pattern_char_comparison
[`manual_range_contains`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_range_contains
[`manual_range_patterns`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_range_patterns
[`manual_rem_euclid`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_rem_euclid
[`manual_repeat_n`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_repeat_n
[`manual_retain`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_retain
[`manual_rotate`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_rotate
[`manual_saturating_arithmetic`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_saturating_arithmetic
[`manual_slice_fill`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_slice_fill
[`manual_slice_size_calculation`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_slice_size_calculation
[`manual_split_once`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_split_once
[`manual_str_repeat`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_str_repeat
[`manual_string_new`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_string_new
[`manual_strip`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_strip
[`manual_swap`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_swap
[`manual_try_fold`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_try_fold
[`manual_unwrap_or`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_unwrap_or
[`manual_unwrap_or_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_unwrap_or_default
[`manual_while_let_some`]: https://rust-lang.github.io/rust-clippy/master/index.html#manual_while_let_some
[`many_single_char_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#many_single_char_names
[`map_all_any_identity`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_all_any_identity
[`map_clone`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_clone
[`map_collect_result_unit`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_collect_result_unit
[`map_entry`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_entry
[`map_err_ignore`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_err_ignore
[`map_flatten`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_flatten
[`map_identity`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_identity
[`map_unwrap_or`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_unwrap_or
[`map_with_unused_argument_over_ranges`]: https://rust-lang.github.io/rust-clippy/master/index.html#map_with_unused_argument_over_ranges
[`match_as_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_as_ref
[`match_bool`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_bool
[`match_like_matches_macro`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_like_matches_macro
[`match_on_vec_items`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_on_vec_items
[`match_overlapping_arm`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_overlapping_arm
[`match_ref_pats`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_ref_pats
[`match_result_ok`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_result_ok
[`match_same_arms`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_same_arms
[`match_single_binding`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_single_binding
[`match_str_case_mismatch`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_str_case_mismatch
[`match_wild_err_arm`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_wild_err_arm
[`match_wildcard_for_single_variants`]: https://rust-lang.github.io/rust-clippy/master/index.html#match_wildcard_for_single_variants
[`maybe_infinite_iter`]: https://rust-lang.github.io/rust-clippy/master/index.html#maybe_infinite_iter
[`maybe_misused_cfg`]: https://rust-lang.github.io/rust-clippy/master/index.html#maybe_misused_cfg
[`mem_discriminant_non_enum`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_discriminant_non_enum
[`mem_forget`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_forget
[`mem_replace_option_with_none`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_replace_option_with_none
[`mem_replace_option_with_some`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_replace_option_with_some
[`mem_replace_with_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_replace_with_default
[`mem_replace_with_uninit`]: https://rust-lang.github.io/rust-clippy/master/index.html#mem_replace_with_uninit
[`min_ident_chars`]: https://rust-lang.github.io/rust-clippy/master/index.html#min_ident_chars
[`min_max`]: https://rust-lang.github.io/rust-clippy/master/index.html#min_max
[`misaligned_transmute`]: https://rust-lang.github.io/rust-clippy/master/index.html#misaligned_transmute
[`mismatched_target_os`]: https://rust-lang.github.io/rust-clippy/master/index.html#mismatched_target_os
[`mismatching_type_param_order`]: https://rust-lang.github.io/rust-clippy/master/index.html#mismatching_type_param_order
[`misnamed_getters`]: https://rust-lang.github.io/rust-clippy/master/index.html#misnamed_getters
[`misrefactored_assign_op`]: https://rust-lang.github.io/rust-clippy/master/index.html#misrefactored_assign_op
[`missing_assert_message`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_assert_message
[`missing_asserts_for_indexing`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_asserts_for_indexing
[`missing_const_for_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_const_for_fn
[`missing_const_for_thread_local`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_const_for_thread_local
[`missing_docs_in_private_items`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_docs_in_private_items
[`missing_enforced_import_renames`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_enforced_import_renames
[`missing_errors_doc`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_errors_doc
[`missing_fields_in_debug`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_fields_in_debug
[`missing_inline_in_public_items`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_inline_in_public_items
[`missing_panics_doc`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_panics_doc
[`missing_safety_doc`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_safety_doc
[`missing_spin_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_spin_loop
[`missing_trait_methods`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_trait_methods
[`missing_transmute_annotations`]: https://rust-lang.github.io/rust-clippy/master/index.html#missing_transmute_annotations
[`mistyped_literal_suffixes`]: https://rust-lang.github.io/rust-clippy/master/index.html#mistyped_literal_suffixes
[`mixed_attributes_style`]: https://rust-lang.github.io/rust-clippy/master/index.html#mixed_attributes_style
[`mixed_case_hex_literals`]: https://rust-lang.github.io/rust-clippy/master/index.html#mixed_case_hex_literals
[`mixed_read_write_in_expression`]: https://rust-lang.github.io/rust-clippy/master/index.html#mixed_read_write_in_expression
[`mod_module_files`]: https://rust-lang.github.io/rust-clippy/master/index.html#mod_module_files
[`module_inception`]: https://rust-lang.github.io/rust-clippy/master/index.html#module_inception
[`module_name_repetitions`]: https://rust-lang.github.io/rust-clippy/master/index.html#module_name_repetitions
[`modulo_arithmetic`]: https://rust-lang.github.io/rust-clippy/master/index.html#modulo_arithmetic
[`modulo_one`]: https://rust-lang.github.io/rust-clippy/master/index.html#modulo_one
[`multi_assignments`]: https://rust-lang.github.io/rust-clippy/master/index.html#multi_assignments
[`multiple_bound_locations`]: https://rust-lang.github.io/rust-clippy/master/index.html#multiple_bound_locations
[`multiple_crate_versions`]: https://rust-lang.github.io/rust-clippy/master/index.html#multiple_crate_versions
[`multiple_inherent_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#multiple_inherent_impl
[`multiple_unsafe_ops_per_block`]: https://rust-lang.github.io/rust-clippy/master/index.html#multiple_unsafe_ops_per_block
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
[`needless_as_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_as_bytes
[`needless_bitwise_bool`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_bitwise_bool
[`needless_bool`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_bool
[`needless_bool_assign`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_bool_assign
[`needless_borrow`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_borrow
[`needless_borrowed_reference`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_borrowed_reference
[`needless_borrows_for_generic_args`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_borrows_for_generic_args
[`needless_character_iteration`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_character_iteration
[`needless_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_collect
[`needless_continue`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_continue
[`needless_doctest_main`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_doctest_main
[`needless_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_else
[`needless_for_each`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_for_each
[`needless_if`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_if
[`needless_late_init`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_late_init
[`needless_lifetimes`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_lifetimes
[`needless_match`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_match
[`needless_maybe_sized`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_maybe_sized
[`needless_option_as_deref`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_option_as_deref
[`needless_option_take`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_option_take
[`needless_parens_on_range_literals`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_parens_on_range_literals
[`needless_pass_by_ref_mut`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_pass_by_ref_mut
[`needless_pass_by_value`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_pass_by_value
[`needless_pub_self`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_pub_self
[`needless_question_mark`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_question_mark
[`needless_range_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_range_loop
[`needless_raw_string_hashes`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_raw_string_hashes
[`needless_raw_strings`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_raw_strings
[`needless_return`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_return
[`needless_return_with_question_mark`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_return_with_question_mark
[`needless_splitn`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_splitn
[`needless_update`]: https://rust-lang.github.io/rust-clippy/master/index.html#needless_update
[`neg_cmp_op_on_partial_ord`]: https://rust-lang.github.io/rust-clippy/master/index.html#neg_cmp_op_on_partial_ord
[`neg_multiply`]: https://rust-lang.github.io/rust-clippy/master/index.html#neg_multiply
[`negative_feature_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#negative_feature_names
[`never_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#never_loop
[`new_ret_no_self`]: https://rust-lang.github.io/rust-clippy/master/index.html#new_ret_no_self
[`new_without_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#new_without_default
[`new_without_default_derive`]: https://rust-lang.github.io/rust-clippy/master/index.html#new_without_default_derive
[`no_effect`]: https://rust-lang.github.io/rust-clippy/master/index.html#no_effect
[`no_effect_replace`]: https://rust-lang.github.io/rust-clippy/master/index.html#no_effect_replace
[`no_effect_underscore_binding`]: https://rust-lang.github.io/rust-clippy/master/index.html#no_effect_underscore_binding
[`no_mangle_with_rust_abi`]: https://rust-lang.github.io/rust-clippy/master/index.html#no_mangle_with_rust_abi
[`non_ascii_literal`]: https://rust-lang.github.io/rust-clippy/master/index.html#non_ascii_literal
[`non_canonical_clone_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#non_canonical_clone_impl
[`non_canonical_partial_ord_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#non_canonical_partial_ord_impl
[`non_minimal_cfg`]: https://rust-lang.github.io/rust-clippy/master/index.html#non_minimal_cfg
[`non_octal_unix_permissions`]: https://rust-lang.github.io/rust-clippy/master/index.html#non_octal_unix_permissions
[`non_send_fields_in_send_ty`]: https://rust-lang.github.io/rust-clippy/master/index.html#non_send_fields_in_send_ty
[`non_std_lazy_statics`]: https://rust-lang.github.io/rust-clippy/master/index.html#non_std_lazy_statics
[`non_zero_suggestions`]: https://rust-lang.github.io/rust-clippy/master/index.html#non_zero_suggestions
[`nonminimal_bool`]: https://rust-lang.github.io/rust-clippy/master/index.html#nonminimal_bool
[`nonsensical_open_options`]: https://rust-lang.github.io/rust-clippy/master/index.html#nonsensical_open_options
[`nonstandard_macro_braces`]: https://rust-lang.github.io/rust-clippy/master/index.html#nonstandard_macro_braces
[`not_unsafe_ptr_arg_deref`]: https://rust-lang.github.io/rust-clippy/master/index.html#not_unsafe_ptr_arg_deref
[`obfuscated_if_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#obfuscated_if_else
[`octal_escapes`]: https://rust-lang.github.io/rust-clippy/master/index.html#octal_escapes
[`ok_expect`]: https://rust-lang.github.io/rust-clippy/master/index.html#ok_expect
[`only_used_in_recursion`]: https://rust-lang.github.io/rust-clippy/master/index.html#only_used_in_recursion
[`op_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#op_ref
[`option_and_then_some`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_and_then_some
[`option_as_ref_cloned`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_as_ref_cloned
[`option_as_ref_deref`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_as_ref_deref
[`option_env_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_env_unwrap
[`option_expect_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_expect_used
[`option_filter_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_filter_map
[`option_if_let_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_if_let_else
[`option_map_or_err_ok`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_map_or_err_ok
[`option_map_or_none`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_map_or_none
[`option_map_unit_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_map_unit_fn
[`option_map_unwrap_or`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_map_unwrap_or
[`option_map_unwrap_or_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_map_unwrap_or_else
[`option_option`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_option
[`option_unwrap_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#option_unwrap_used
[`or_fun_call`]: https://rust-lang.github.io/rust-clippy/master/index.html#or_fun_call
[`or_then_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#or_then_unwrap
[`out_of_bounds_indexing`]: https://rust-lang.github.io/rust-clippy/master/index.html#out_of_bounds_indexing
[`overflow_check_conditional`]: https://rust-lang.github.io/rust-clippy/master/index.html#overflow_check_conditional
[`overly_complex_bool_expr`]: https://rust-lang.github.io/rust-clippy/master/index.html#overly_complex_bool_expr
[`owned_cow`]: https://rust-lang.github.io/rust-clippy/master/index.html#owned_cow
[`panic`]: https://rust-lang.github.io/rust-clippy/master/index.html#panic
[`panic_in_result_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#panic_in_result_fn
[`panic_params`]: https://rust-lang.github.io/rust-clippy/master/index.html#panic_params
[`panicking_overflow_checks`]: https://rust-lang.github.io/rust-clippy/master/index.html#panicking_overflow_checks
[`panicking_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#panicking_unwrap
[`partial_pub_fields`]: https://rust-lang.github.io/rust-clippy/master/index.html#partial_pub_fields
[`partialeq_ne_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#partialeq_ne_impl
[`partialeq_to_none`]: https://rust-lang.github.io/rust-clippy/master/index.html#partialeq_to_none
[`path_buf_push_overwrite`]: https://rust-lang.github.io/rust-clippy/master/index.html#path_buf_push_overwrite
[`path_ends_with_ext`]: https://rust-lang.github.io/rust-clippy/master/index.html#path_ends_with_ext
[`pathbuf_init_then_push`]: https://rust-lang.github.io/rust-clippy/master/index.html#pathbuf_init_then_push
[`pattern_type_mismatch`]: https://rust-lang.github.io/rust-clippy/master/index.html#pattern_type_mismatch
[`permissions_set_readonly_false`]: https://rust-lang.github.io/rust-clippy/master/index.html#permissions_set_readonly_false
[`pointers_in_nomem_asm_block`]: https://rust-lang.github.io/rust-clippy/master/index.html#pointers_in_nomem_asm_block
[`positional_named_format_parameters`]: https://rust-lang.github.io/rust-clippy/master/index.html#positional_named_format_parameters
[`possible_missing_comma`]: https://rust-lang.github.io/rust-clippy/master/index.html#possible_missing_comma
[`precedence`]: https://rust-lang.github.io/rust-clippy/master/index.html#precedence
[`precedence_bits`]: https://rust-lang.github.io/rust-clippy/master/index.html#precedence_bits
[`print_in_format_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#print_in_format_impl
[`print_literal`]: https://rust-lang.github.io/rust-clippy/master/index.html#print_literal
[`print_stderr`]: https://rust-lang.github.io/rust-clippy/master/index.html#print_stderr
[`print_stdout`]: https://rust-lang.github.io/rust-clippy/master/index.html#print_stdout
[`print_with_newline`]: https://rust-lang.github.io/rust-clippy/master/index.html#print_with_newline
[`println_empty_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#println_empty_string
[`ptr_arg`]: https://rust-lang.github.io/rust-clippy/master/index.html#ptr_arg
[`ptr_as_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#ptr_as_ptr
[`ptr_cast_constness`]: https://rust-lang.github.io/rust-clippy/master/index.html#ptr_cast_constness
[`ptr_eq`]: https://rust-lang.github.io/rust-clippy/master/index.html#ptr_eq
[`ptr_offset_with_cast`]: https://rust-lang.github.io/rust-clippy/master/index.html#ptr_offset_with_cast
[`pub_enum_variant_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#pub_enum_variant_names
[`pub_underscore_fields`]: https://rust-lang.github.io/rust-clippy/master/index.html#pub_underscore_fields
[`pub_use`]: https://rust-lang.github.io/rust-clippy/master/index.html#pub_use
[`pub_with_shorthand`]: https://rust-lang.github.io/rust-clippy/master/index.html#pub_with_shorthand
[`pub_without_shorthand`]: https://rust-lang.github.io/rust-clippy/master/index.html#pub_without_shorthand
[`question_mark`]: https://rust-lang.github.io/rust-clippy/master/index.html#question_mark
[`question_mark_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#question_mark_used
[`range_minus_one`]: https://rust-lang.github.io/rust-clippy/master/index.html#range_minus_one
[`range_plus_one`]: https://rust-lang.github.io/rust-clippy/master/index.html#range_plus_one
[`range_step_by_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#range_step_by_zero
[`range_zip_with_len`]: https://rust-lang.github.io/rust-clippy/master/index.html#range_zip_with_len
[`rc_buffer`]: https://rust-lang.github.io/rust-clippy/master/index.html#rc_buffer
[`rc_clone_in_vec_init`]: https://rust-lang.github.io/rust-clippy/master/index.html#rc_clone_in_vec_init
[`rc_mutex`]: https://rust-lang.github.io/rust-clippy/master/index.html#rc_mutex
[`read_line_without_trim`]: https://rust-lang.github.io/rust-clippy/master/index.html#read_line_without_trim
[`read_zero_byte_vec`]: https://rust-lang.github.io/rust-clippy/master/index.html#read_zero_byte_vec
[`readonly_write_lock`]: https://rust-lang.github.io/rust-clippy/master/index.html#readonly_write_lock
[`recursive_format_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#recursive_format_impl
[`redundant_allocation`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_allocation
[`redundant_as_str`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_as_str
[`redundant_async_block`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_async_block
[`redundant_at_rest_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_at_rest_pattern
[`redundant_clone`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_clone
[`redundant_closure`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_closure
[`redundant_closure_call`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_closure_call
[`redundant_closure_for_method_calls`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_closure_for_method_calls
[`redundant_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_comparisons
[`redundant_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_else
[`redundant_feature_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_feature_names
[`redundant_field_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_field_names
[`redundant_guards`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_guards
[`redundant_locals`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_locals
[`redundant_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_pattern
[`redundant_pattern_matching`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_pattern_matching
[`redundant_pub_crate`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_pub_crate
[`redundant_slicing`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_slicing
[`redundant_static_lifetimes`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_static_lifetimes
[`redundant_test_prefix`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_test_prefix
[`redundant_type_annotations`]: https://rust-lang.github.io/rust-clippy/master/index.html#redundant_type_annotations
[`ref_as_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#ref_as_ptr
[`ref_binding_to_reference`]: https://rust-lang.github.io/rust-clippy/master/index.html#ref_binding_to_reference
[`ref_in_deref`]: https://rust-lang.github.io/rust-clippy/master/index.html#ref_in_deref
[`ref_option`]: https://rust-lang.github.io/rust-clippy/master/index.html#ref_option
[`ref_option_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#ref_option_ref
[`ref_patterns`]: https://rust-lang.github.io/rust-clippy/master/index.html#ref_patterns
[`regex_creation_in_loops`]: https://rust-lang.github.io/rust-clippy/master/index.html#regex_creation_in_loops
[`regex_macro`]: https://rust-lang.github.io/rust-clippy/master/index.html#regex_macro
[`renamed_function_params`]: https://rust-lang.github.io/rust-clippy/master/index.html#renamed_function_params
[`repeat_once`]: https://rust-lang.github.io/rust-clippy/master/index.html#repeat_once
[`repeat_vec_with_capacity`]: https://rust-lang.github.io/rust-clippy/master/index.html#repeat_vec_with_capacity
[`replace_consts`]: https://rust-lang.github.io/rust-clippy/master/index.html#replace_consts
[`repr_packed_without_abi`]: https://rust-lang.github.io/rust-clippy/master/index.html#repr_packed_without_abi
[`reserve_after_initialization`]: https://rust-lang.github.io/rust-clippy/master/index.html#reserve_after_initialization
[`rest_pat_in_fully_bound_structs`]: https://rust-lang.github.io/rust-clippy/master/index.html#rest_pat_in_fully_bound_structs
[`result_expect_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_expect_used
[`result_filter_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_filter_map
[`result_large_err`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_large_err
[`result_map_or_into_option`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_map_or_into_option
[`result_map_unit_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_map_unit_fn
[`result_map_unwrap_or_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_map_unwrap_or_else
[`result_unit_err`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_unit_err
[`result_unwrap_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#result_unwrap_used
[`return_and_then`]: https://rust-lang.github.io/rust-clippy/master/index.html#return_and_then
[`return_self_not_must_use`]: https://rust-lang.github.io/rust-clippy/master/index.html#return_self_not_must_use
[`reverse_range_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#reverse_range_loop
[`reversed_empty_ranges`]: https://rust-lang.github.io/rust-clippy/master/index.html#reversed_empty_ranges
[`same_functions_in_if_condition`]: https://rust-lang.github.io/rust-clippy/master/index.html#same_functions_in_if_condition
[`same_item_push`]: https://rust-lang.github.io/rust-clippy/master/index.html#same_item_push
[`same_name_method`]: https://rust-lang.github.io/rust-clippy/master/index.html#same_name_method
[`search_is_some`]: https://rust-lang.github.io/rust-clippy/master/index.html#search_is_some
[`seek_from_current`]: https://rust-lang.github.io/rust-clippy/master/index.html#seek_from_current
[`seek_to_start_instead_of_rewind`]: https://rust-lang.github.io/rust-clippy/master/index.html#seek_to_start_instead_of_rewind
[`self_assignment`]: https://rust-lang.github.io/rust-clippy/master/index.html#self_assignment
[`self_named_constructors`]: https://rust-lang.github.io/rust-clippy/master/index.html#self_named_constructors
[`self_named_module_files`]: https://rust-lang.github.io/rust-clippy/master/index.html#self_named_module_files
[`semicolon_if_nothing_returned`]: https://rust-lang.github.io/rust-clippy/master/index.html#semicolon_if_nothing_returned
[`semicolon_inside_block`]: https://rust-lang.github.io/rust-clippy/master/index.html#semicolon_inside_block
[`semicolon_outside_block`]: https://rust-lang.github.io/rust-clippy/master/index.html#semicolon_outside_block
[`separated_literal_suffix`]: https://rust-lang.github.io/rust-clippy/master/index.html#separated_literal_suffix
[`serde_api_misuse`]: https://rust-lang.github.io/rust-clippy/master/index.html#serde_api_misuse
[`set_contains_or_insert`]: https://rust-lang.github.io/rust-clippy/master/index.html#set_contains_or_insert
[`shadow_reuse`]: https://rust-lang.github.io/rust-clippy/master/index.html#shadow_reuse
[`shadow_same`]: https://rust-lang.github.io/rust-clippy/master/index.html#shadow_same
[`shadow_unrelated`]: https://rust-lang.github.io/rust-clippy/master/index.html#shadow_unrelated
[`short_circuit_statement`]: https://rust-lang.github.io/rust-clippy/master/index.html#short_circuit_statement
[`should_assert_eq`]: https://rust-lang.github.io/rust-clippy/master/index.html#should_assert_eq
[`should_implement_trait`]: https://rust-lang.github.io/rust-clippy/master/index.html#should_implement_trait
[`should_panic_without_expect`]: https://rust-lang.github.io/rust-clippy/master/index.html#should_panic_without_expect
[`significant_drop_in_scrutinee`]: https://rust-lang.github.io/rust-clippy/master/index.html#significant_drop_in_scrutinee
[`significant_drop_tightening`]: https://rust-lang.github.io/rust-clippy/master/index.html#significant_drop_tightening
[`similar_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#similar_names
[`single_call_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_call_fn
[`single_char_add_str`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_char_add_str
[`single_char_lifetime_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_char_lifetime_names
[`single_char_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_char_pattern
[`single_char_push_str`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_char_push_str
[`single_component_path_imports`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_component_path_imports
[`single_element_loop`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_element_loop
[`single_match`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_match
[`single_match_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_match_else
[`single_option_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_option_map
[`single_range_in_vec_init`]: https://rust-lang.github.io/rust-clippy/master/index.html#single_range_in_vec_init
[`size_of_in_element_count`]: https://rust-lang.github.io/rust-clippy/master/index.html#size_of_in_element_count
[`size_of_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#size_of_ref
[`skip_while_next`]: https://rust-lang.github.io/rust-clippy/master/index.html#skip_while_next
[`sliced_string_as_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#sliced_string_as_bytes
[`slow_vector_initialization`]: https://rust-lang.github.io/rust-clippy/master/index.html#slow_vector_initialization
[`stable_sort_primitive`]: https://rust-lang.github.io/rust-clippy/master/index.html#stable_sort_primitive
[`std_instead_of_alloc`]: https://rust-lang.github.io/rust-clippy/master/index.html#std_instead_of_alloc
[`std_instead_of_core`]: https://rust-lang.github.io/rust-clippy/master/index.html#std_instead_of_core
[`str_split_at_newline`]: https://rust-lang.github.io/rust-clippy/master/index.html#str_split_at_newline
[`str_to_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#str_to_string
[`string_add`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_add
[`string_add_assign`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_add_assign
[`string_extend_chars`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_extend_chars
[`string_from_utf8_as_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_from_utf8_as_bytes
[`string_lit_as_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_lit_as_bytes
[`string_lit_chars_any`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_lit_chars_any
[`string_slice`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_slice
[`string_to_string`]: https://rust-lang.github.io/rust-clippy/master/index.html#string_to_string
[`strlen_on_c_strings`]: https://rust-lang.github.io/rust-clippy/master/index.html#strlen_on_c_strings
[`struct_excessive_bools`]: https://rust-lang.github.io/rust-clippy/master/index.html#struct_excessive_bools
[`struct_field_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#struct_field_names
[`stutter`]: https://rust-lang.github.io/rust-clippy/master/index.html#stutter
[`suboptimal_flops`]: https://rust-lang.github.io/rust-clippy/master/index.html#suboptimal_flops
[`suspicious_arithmetic_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_arithmetic_impl
[`suspicious_assignment_formatting`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_assignment_formatting
[`suspicious_command_arg_space`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_command_arg_space
[`suspicious_doc_comments`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_doc_comments
[`suspicious_else_formatting`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_else_formatting
[`suspicious_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_map
[`suspicious_op_assign_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_op_assign_impl
[`suspicious_open_options`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_open_options
[`suspicious_operation_groupings`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_operation_groupings
[`suspicious_splitn`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_splitn
[`suspicious_to_owned`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_to_owned
[`suspicious_unary_op_formatting`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_unary_op_formatting
[`suspicious_xor_used_as_pow`]: https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_xor_used_as_pow
[`swap_ptr_to_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#swap_ptr_to_ref
[`swap_with_temporary`]: https://rust-lang.github.io/rust-clippy/master/index.html#swap_with_temporary
[`tabs_in_doc_comments`]: https://rust-lang.github.io/rust-clippy/master/index.html#tabs_in_doc_comments
[`temporary_assignment`]: https://rust-lang.github.io/rust-clippy/master/index.html#temporary_assignment
[`temporary_cstring_as_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#temporary_cstring_as_ptr
[`test_attr_in_doctest`]: https://rust-lang.github.io/rust-clippy/master/index.html#test_attr_in_doctest
[`tests_outside_test_module`]: https://rust-lang.github.io/rust-clippy/master/index.html#tests_outside_test_module
[`thread_local_initializer_can_be_made_const`]: https://rust-lang.github.io/rust-clippy/master/index.html#thread_local_initializer_can_be_made_const
[`to_digit_is_some`]: https://rust-lang.github.io/rust-clippy/master/index.html#to_digit_is_some
[`to_string_in_display`]: https://rust-lang.github.io/rust-clippy/master/index.html#to_string_in_display
[`to_string_in_format_args`]: https://rust-lang.github.io/rust-clippy/master/index.html#to_string_in_format_args
[`to_string_trait_impl`]: https://rust-lang.github.io/rust-clippy/master/index.html#to_string_trait_impl
[`todo`]: https://rust-lang.github.io/rust-clippy/master/index.html#todo
[`too_long_first_doc_paragraph`]: https://rust-lang.github.io/rust-clippy/master/index.html#too_long_first_doc_paragraph
[`too_many_arguments`]: https://rust-lang.github.io/rust-clippy/master/index.html#too_many_arguments
[`too_many_lines`]: https://rust-lang.github.io/rust-clippy/master/index.html#too_many_lines
[`toplevel_ref_arg`]: https://rust-lang.github.io/rust-clippy/master/index.html#toplevel_ref_arg
[`trailing_empty_array`]: https://rust-lang.github.io/rust-clippy/master/index.html#trailing_empty_array
[`trait_duplication_in_bounds`]: https://rust-lang.github.io/rust-clippy/master/index.html#trait_duplication_in_bounds
[`transmute_bytes_to_str`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_bytes_to_str
[`transmute_float_to_int`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_float_to_int
[`transmute_int_to_bool`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_int_to_bool
[`transmute_int_to_char`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_int_to_char
[`transmute_int_to_float`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_int_to_float
[`transmute_int_to_non_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_int_to_non_zero
[`transmute_null_to_fn`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_null_to_fn
[`transmute_num_to_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_num_to_bytes
[`transmute_ptr_to_ptr`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_ptr_to_ptr
[`transmute_ptr_to_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_ptr_to_ref
[`transmute_undefined_repr`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmute_undefined_repr
[`transmutes_expressible_as_ptr_casts`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmutes_expressible_as_ptr_casts
[`transmuting_null`]: https://rust-lang.github.io/rust-clippy/master/index.html#transmuting_null
[`trim_split_whitespace`]: https://rust-lang.github.io/rust-clippy/master/index.html#trim_split_whitespace
[`trivial_regex`]: https://rust-lang.github.io/rust-clippy/master/index.html#trivial_regex
[`trivially_copy_pass_by_ref`]: https://rust-lang.github.io/rust-clippy/master/index.html#trivially_copy_pass_by_ref
[`try_err`]: https://rust-lang.github.io/rust-clippy/master/index.html#try_err
[`tuple_array_conversions`]: https://rust-lang.github.io/rust-clippy/master/index.html#tuple_array_conversions
[`type_complexity`]: https://rust-lang.github.io/rust-clippy/master/index.html#type_complexity
[`type_id_on_box`]: https://rust-lang.github.io/rust-clippy/master/index.html#type_id_on_box
[`type_repetition_in_bounds`]: https://rust-lang.github.io/rust-clippy/master/index.html#type_repetition_in_bounds
[`unbuffered_bytes`]: https://rust-lang.github.io/rust-clippy/master/index.html#unbuffered_bytes
[`unchecked_duration_subtraction`]: https://rust-lang.github.io/rust-clippy/master/index.html#unchecked_duration_subtraction
[`unconditional_recursion`]: https://rust-lang.github.io/rust-clippy/master/index.html#unconditional_recursion
[`undocumented_unsafe_blocks`]: https://rust-lang.github.io/rust-clippy/master/index.html#undocumented_unsafe_blocks
[`undropped_manually_drops`]: https://rust-lang.github.io/rust-clippy/master/index.html#undropped_manually_drops
[`unicode_not_nfc`]: https://rust-lang.github.io/rust-clippy/master/index.html#unicode_not_nfc
[`unimplemented`]: https://rust-lang.github.io/rust-clippy/master/index.html#unimplemented
[`uninhabited_references`]: https://rust-lang.github.io/rust-clippy/master/index.html#uninhabited_references
[`uninit_assumed_init`]: https://rust-lang.github.io/rust-clippy/master/index.html#uninit_assumed_init
[`uninit_vec`]: https://rust-lang.github.io/rust-clippy/master/index.html#uninit_vec
[`uninlined_format_args`]: https://rust-lang.github.io/rust-clippy/master/index.html#uninlined_format_args
[`unit_arg`]: https://rust-lang.github.io/rust-clippy/master/index.html#unit_arg
[`unit_cmp`]: https://rust-lang.github.io/rust-clippy/master/index.html#unit_cmp
[`unit_hash`]: https://rust-lang.github.io/rust-clippy/master/index.html#unit_hash
[`unit_return_expecting_ord`]: https://rust-lang.github.io/rust-clippy/master/index.html#unit_return_expecting_ord
[`unknown_clippy_lints`]: https://rust-lang.github.io/rust-clippy/master/index.html#unknown_clippy_lints
[`unnecessary_box_returns`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_box_returns
[`unnecessary_cast`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_cast
[`unnecessary_clippy_cfg`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_clippy_cfg
[`unnecessary_debug_formatting`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_debug_formatting
[`unnecessary_fallible_conversions`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_fallible_conversions
[`unnecessary_filter_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_filter_map
[`unnecessary_find_map`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_find_map
[`unnecessary_first_then_check`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_first_then_check
[`unnecessary_fold`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_fold
[`unnecessary_get_then_check`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_get_then_check
[`unnecessary_join`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_join
[`unnecessary_lazy_evaluations`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_lazy_evaluations
[`unnecessary_literal_bound`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_literal_bound
[`unnecessary_literal_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_literal_unwrap
[`unnecessary_map_on_constructor`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_map_on_constructor
[`unnecessary_map_or`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_map_or
[`unnecessary_min_or_max`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_min_or_max
[`unnecessary_mut_passed`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_mut_passed
[`unnecessary_operation`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_operation
[`unnecessary_owned_empty_strings`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_owned_empty_strings
[`unnecessary_result_map_or_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_result_map_or_else
[`unnecessary_safety_comment`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_safety_comment
[`unnecessary_safety_doc`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_safety_doc
[`unnecessary_self_imports`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_self_imports
[`unnecessary_semicolon`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_semicolon
[`unnecessary_sort_by`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_sort_by
[`unnecessary_struct_initialization`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_struct_initialization
[`unnecessary_to_owned`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_to_owned
[`unnecessary_unwrap`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_unwrap
[`unnecessary_wraps`]: https://rust-lang.github.io/rust-clippy/master/index.html#unnecessary_wraps
[`unneeded_field_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#unneeded_field_pattern
[`unneeded_struct_pattern`]: https://rust-lang.github.io/rust-clippy/master/index.html#unneeded_struct_pattern
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
[`unused_async`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_async
[`unused_collect`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_collect
[`unused_enumerate_index`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_enumerate_index
[`unused_format_specs`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_format_specs
[`unused_io_amount`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_io_amount
[`unused_label`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_label
[`unused_peekable`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_peekable
[`unused_result_ok`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_result_ok
[`unused_rounding`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_rounding
[`unused_self`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_self
[`unused_trait_names`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_trait_names
[`unused_unit`]: https://rust-lang.github.io/rust-clippy/master/index.html#unused_unit
[`unusual_byte_groupings`]: https://rust-lang.github.io/rust-clippy/master/index.html#unusual_byte_groupings
[`unwrap_in_result`]: https://rust-lang.github.io/rust-clippy/master/index.html#unwrap_in_result
[`unwrap_or_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#unwrap_or_default
[`unwrap_or_else_default`]: https://rust-lang.github.io/rust-clippy/master/index.html#unwrap_or_else_default
[`unwrap_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#unwrap_used
[`upper_case_acronyms`]: https://rust-lang.github.io/rust-clippy/master/index.html#upper_case_acronyms
[`use_debug`]: https://rust-lang.github.io/rust-clippy/master/index.html#use_debug
[`use_self`]: https://rust-lang.github.io/rust-clippy/master/index.html#use_self
[`used_underscore_binding`]: https://rust-lang.github.io/rust-clippy/master/index.html#used_underscore_binding
[`used_underscore_items`]: https://rust-lang.github.io/rust-clippy/master/index.html#used_underscore_items
[`useless_asref`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_asref
[`useless_attribute`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_attribute
[`useless_conversion`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_conversion
[`useless_format`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_format
[`useless_let_if_seq`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_let_if_seq
[`useless_nonzero_new_unchecked`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_nonzero_new_unchecked
[`useless_transmute`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_transmute
[`useless_vec`]: https://rust-lang.github.io/rust-clippy/master/index.html#useless_vec
[`vec_box`]: https://rust-lang.github.io/rust-clippy/master/index.html#vec_box
[`vec_init_then_push`]: https://rust-lang.github.io/rust-clippy/master/index.html#vec_init_then_push
[`vec_resize_to_zero`]: https://rust-lang.github.io/rust-clippy/master/index.html#vec_resize_to_zero
[`verbose_bit_mask`]: https://rust-lang.github.io/rust-clippy/master/index.html#verbose_bit_mask
[`verbose_file_reads`]: https://rust-lang.github.io/rust-clippy/master/index.html#verbose_file_reads
[`vtable_address_comparisons`]: https://rust-lang.github.io/rust-clippy/master/index.html#vtable_address_comparisons
[`waker_clone_wake`]: https://rust-lang.github.io/rust-clippy/master/index.html#waker_clone_wake
[`while_float`]: https://rust-lang.github.io/rust-clippy/master/index.html#while_float
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
[`zero_repeat_side_effects`]: https://rust-lang.github.io/rust-clippy/master/index.html#zero_repeat_side_effects
[`zero_sized_map_values`]: https://rust-lang.github.io/rust-clippy/master/index.html#zero_sized_map_values
[`zero_width_space`]: https://rust-lang.github.io/rust-clippy/master/index.html#zero_width_space
[`zombie_processes`]: https://rust-lang.github.io/rust-clippy/master/index.html#zombie_processes
[`zst_offset`]: https://rust-lang.github.io/rust-clippy/master/index.html#zst_offset
<!-- end autogenerated links to lint list -->
<!-- begin autogenerated links to configuration documentation -->
[`absolute-paths-allowed-crates`]: https://doc.rust-lang.org/clippy/lint_configuration.html#absolute-paths-allowed-crates
[`absolute-paths-max-segments`]: https://doc.rust-lang.org/clippy/lint_configuration.html#absolute-paths-max-segments
[`accept-comment-above-attributes`]: https://doc.rust-lang.org/clippy/lint_configuration.html#accept-comment-above-attributes
[`accept-comment-above-statement`]: https://doc.rust-lang.org/clippy/lint_configuration.html#accept-comment-above-statement
[`allow-comparison-to-zero`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-comparison-to-zero
[`allow-dbg-in-tests`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-dbg-in-tests
[`allow-exact-repetitions`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-exact-repetitions
[`allow-expect-in-consts`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-expect-in-consts
[`allow-expect-in-tests`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-expect-in-tests
[`allow-indexing-slicing-in-tests`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-indexing-slicing-in-tests
[`allow-mixed-uninlined-format-args`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-mixed-uninlined-format-args
[`allow-one-hash-in-raw-strings`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-one-hash-in-raw-strings
[`allow-panic-in-tests`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-panic-in-tests
[`allow-print-in-tests`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-print-in-tests
[`allow-private-module-inception`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-private-module-inception
[`allow-renamed-params-for`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-renamed-params-for
[`allow-unwrap-in-consts`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-unwrap-in-consts
[`allow-unwrap-in-tests`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-unwrap-in-tests
[`allow-useless-vec-in-tests`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allow-useless-vec-in-tests
[`allowed-dotfiles`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allowed-dotfiles
[`allowed-duplicate-crates`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allowed-duplicate-crates
[`allowed-idents-below-min-chars`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allowed-idents-below-min-chars
[`allowed-prefixes`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allowed-prefixes
[`allowed-scripts`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allowed-scripts
[`allowed-wildcard-imports`]: https://doc.rust-lang.org/clippy/lint_configuration.html#allowed-wildcard-imports
[`arithmetic-side-effects-allowed`]: https://doc.rust-lang.org/clippy/lint_configuration.html#arithmetic-side-effects-allowed
[`arithmetic-side-effects-allowed-binary`]: https://doc.rust-lang.org/clippy/lint_configuration.html#arithmetic-side-effects-allowed-binary
[`arithmetic-side-effects-allowed-unary`]: https://doc.rust-lang.org/clippy/lint_configuration.html#arithmetic-side-effects-allowed-unary
[`array-size-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#array-size-threshold
[`avoid-breaking-exported-api`]: https://doc.rust-lang.org/clippy/lint_configuration.html#avoid-breaking-exported-api
[`await-holding-invalid-types`]: https://doc.rust-lang.org/clippy/lint_configuration.html#await-holding-invalid-types
[`cargo-ignore-publish`]: https://doc.rust-lang.org/clippy/lint_configuration.html#cargo-ignore-publish
[`check-incompatible-msrv-in-tests`]: https://doc.rust-lang.org/clippy/lint_configuration.html#check-incompatible-msrv-in-tests
[`check-inconsistent-struct-field-initializers`]: https://doc.rust-lang.org/clippy/lint_configuration.html#check-inconsistent-struct-field-initializers
[`check-private-items`]: https://doc.rust-lang.org/clippy/lint_configuration.html#check-private-items
[`cognitive-complexity-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#cognitive-complexity-threshold
[`disallowed-macros`]: https://doc.rust-lang.org/clippy/lint_configuration.html#disallowed-macros
[`disallowed-methods`]: https://doc.rust-lang.org/clippy/lint_configuration.html#disallowed-methods
[`disallowed-names`]: https://doc.rust-lang.org/clippy/lint_configuration.html#disallowed-names
[`disallowed-types`]: https://doc.rust-lang.org/clippy/lint_configuration.html#disallowed-types
[`doc-valid-idents`]: https://doc.rust-lang.org/clippy/lint_configuration.html#doc-valid-idents
[`enable-raw-pointer-heuristic-for-send`]: https://doc.rust-lang.org/clippy/lint_configuration.html#enable-raw-pointer-heuristic-for-send
[`enforce-iter-loop-reborrow`]: https://doc.rust-lang.org/clippy/lint_configuration.html#enforce-iter-loop-reborrow
[`enforced-import-renames`]: https://doc.rust-lang.org/clippy/lint_configuration.html#enforced-import-renames
[`enum-variant-name-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#enum-variant-name-threshold
[`enum-variant-size-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#enum-variant-size-threshold
[`excessive-nesting-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#excessive-nesting-threshold
[`future-size-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#future-size-threshold
[`ignore-interior-mutability`]: https://doc.rust-lang.org/clippy/lint_configuration.html#ignore-interior-mutability
[`large-error-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#large-error-threshold
[`lint-commented-code`]: https://doc.rust-lang.org/clippy/lint_configuration.html#lint-commented-code
[`literal-representation-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#literal-representation-threshold
[`matches-for-let-else`]: https://doc.rust-lang.org/clippy/lint_configuration.html#matches-for-let-else
[`max-fn-params-bools`]: https://doc.rust-lang.org/clippy/lint_configuration.html#max-fn-params-bools
[`max-include-file-size`]: https://doc.rust-lang.org/clippy/lint_configuration.html#max-include-file-size
[`max-struct-bools`]: https://doc.rust-lang.org/clippy/lint_configuration.html#max-struct-bools
[`max-suggested-slice-pattern-length`]: https://doc.rust-lang.org/clippy/lint_configuration.html#max-suggested-slice-pattern-length
[`max-trait-bounds`]: https://doc.rust-lang.org/clippy/lint_configuration.html#max-trait-bounds
[`min-ident-chars-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#min-ident-chars-threshold
[`missing-docs-allow-unused`]: https://doc.rust-lang.org/clippy/lint_configuration.html#missing-docs-allow-unused
[`missing-docs-in-crate-items`]: https://doc.rust-lang.org/clippy/lint_configuration.html#missing-docs-in-crate-items
[`module-item-order-groupings`]: https://doc.rust-lang.org/clippy/lint_configuration.html#module-item-order-groupings
[`module-items-ordered-within-groupings`]: https://doc.rust-lang.org/clippy/lint_configuration.html#module-items-ordered-within-groupings
[`msrv`]: https://doc.rust-lang.org/clippy/lint_configuration.html#msrv
[`pass-by-value-size-limit`]: https://doc.rust-lang.org/clippy/lint_configuration.html#pass-by-value-size-limit
[`pub-underscore-fields-behavior`]: https://doc.rust-lang.org/clippy/lint_configuration.html#pub-underscore-fields-behavior
[`semicolon-inside-block-ignore-singleline`]: https://doc.rust-lang.org/clippy/lint_configuration.html#semicolon-inside-block-ignore-singleline
[`semicolon-outside-block-ignore-multiline`]: https://doc.rust-lang.org/clippy/lint_configuration.html#semicolon-outside-block-ignore-multiline
[`single-char-binding-names-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#single-char-binding-names-threshold
[`source-item-ordering`]: https://doc.rust-lang.org/clippy/lint_configuration.html#source-item-ordering
[`stack-size-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#stack-size-threshold
[`standard-macro-braces`]: https://doc.rust-lang.org/clippy/lint_configuration.html#standard-macro-braces
[`struct-field-name-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#struct-field-name-threshold
[`suppress-restriction-lint-in-const`]: https://doc.rust-lang.org/clippy/lint_configuration.html#suppress-restriction-lint-in-const
[`too-large-for-stack`]: https://doc.rust-lang.org/clippy/lint_configuration.html#too-large-for-stack
[`too-many-arguments-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#too-many-arguments-threshold
[`too-many-lines-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#too-many-lines-threshold
[`trait-assoc-item-kinds-order`]: https://doc.rust-lang.org/clippy/lint_configuration.html#trait-assoc-item-kinds-order
[`trivial-copy-size-limit`]: https://doc.rust-lang.org/clippy/lint_configuration.html#trivial-copy-size-limit
[`type-complexity-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#type-complexity-threshold
[`unnecessary-box-size`]: https://doc.rust-lang.org/clippy/lint_configuration.html#unnecessary-box-size
[`unreadable-literal-lint-fractions`]: https://doc.rust-lang.org/clippy/lint_configuration.html#unreadable-literal-lint-fractions
[`upper-case-acronyms-aggressive`]: https://doc.rust-lang.org/clippy/lint_configuration.html#upper-case-acronyms-aggressive
[`vec-box-size-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#vec-box-size-threshold
[`verbose-bit-mask-threshold`]: https://doc.rust-lang.org/clippy/lint_configuration.html#verbose-bit-mask-threshold
[`warn-on-all-wildcard-imports`]: https://doc.rust-lang.org/clippy/lint_configuration.html#warn-on-all-wildcard-imports
[`warn-unsafe-macro-metavars-in-private-macros`]: https://doc.rust-lang.org/clippy/lint_configuration.html#warn-unsafe-macro-metavars-in-private-macros
<!-- end autogenerated links to configuration documentation -->
