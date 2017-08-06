# Change Log
All notable changes to this project will be documented in this file.

## 0.0.148
* Update to *rustc 1.21.0-nightly (37c7d0ebb 2017-07-31)*
* New lints: [`unreadable_literal`], [`inconsisten_digit_grouping`], [`large_digit_groups`]

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
* [`option_map_unwrap_or`] and [`option_map_unwrap_or_else`] are now
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
  lint groups: [`filter_next`], [`for_loop_over_option`],
  [`for_loop_over_result`] and [`match_overlapping_arm`]. You should now be
  able to `#[allow/deny]` them individually and they are available directly
  through [`cargo clippy`].

## 0.0.87 — 2016-08-31
* Rustup to *rustc 1.13.0-nightly (eac41469d 2016-08-30)*
* New lints: [`builtin_type_shadow`]
* Fix FP in [`zero_prefixed_literal`] and `0b`/`0o`

## 0.0.86 — 2016-08-28
* Rustup to *rustc 1.13.0-nightly (a23064af5 2016-08-27)*
* New lints: [`missing_docs_in_private_items`], [`zero_prefixed_literal`]

## 0.0.85 — 2016-08-19
* Fix ICE with [`useless_attribute`]
* [`useless_attribute`] ignores [`unused_imports`] on `use` statements

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
* New lints: [`stutter`] and [`iter_nth`]

## 0.0.76 — 2016-06-10
* Rustup to *rustc 1.11.0-nightly (7d2f75a95 2016-06-09)*
* `cargo clippy` now automatically defines the `clippy` feature
* New lint: [`not_unsafe_ptr_arg_deref`]

## 0.0.75 — 2016-06-08
* Rustup to *rustc 1.11.0-nightly (763f9234b 2016-06-06)*

## 0.0.74 — 2016-06-07
* Fix bug with `cargo-clippy` JSON parsing
* Add the `CLIPPY_DISABLE_WIKI_LINKS` environment variable to deactivate the
  “for further information visit *wiki-link*” message.

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

<!-- begin autogenerated links to wiki -->
[`absurd_extreme_comparisons`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#absurd_extreme_comparisons
[`almost_swapped`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#almost_swapped
[`approx_constant`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#approx_constant
[`assign_op_pattern`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#assign_op_pattern
[`assign_ops`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#assign_ops
[`bad_bit_mask`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#bad_bit_mask
[`blacklisted_name`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#blacklisted_name
[`block_in_if_condition_expr`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#block_in_if_condition_expr
[`block_in_if_condition_stmt`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#block_in_if_condition_stmt
[`bool_comparison`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#bool_comparison
[`borrowed_box`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#borrowed_box
[`box_vec`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#box_vec
[`boxed_local`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#boxed_local
[`builtin_type_shadow`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#builtin_type_shadow
[`cast_possible_truncation`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#cast_possible_truncation
[`cast_possible_wrap`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#cast_possible_wrap
[`cast_precision_loss`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#cast_precision_loss
[`cast_sign_loss`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#cast_sign_loss
[`char_lit_as_u8`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#char_lit_as_u8
[`chars_next_cmp`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#chars_next_cmp
[`clone_double_ref`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#clone_double_ref
[`clone_on_copy`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#clone_on_copy
[`cmp_nan`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#cmp_nan
[`cmp_null`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#cmp_null
[`cmp_owned`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#cmp_owned
[`collapsible_if`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#collapsible_if
[`crosspointer_transmute`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#crosspointer_transmute
[`cyclomatic_complexity`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#cyclomatic_complexity
[`deprecated_semver`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#deprecated_semver
[`deref_addrof`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#deref_addrof
[`derive_hash_xor_eq`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#derive_hash_xor_eq
[`diverging_sub_expression`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#diverging_sub_expression
[`doc_markdown`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#doc_markdown
[`double_neg`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#double_neg
[`double_parens`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#double_parens
[`drop_copy`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#drop_copy
[`drop_ref`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#drop_ref
[`duplicate_underscore_argument`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#duplicate_underscore_argument
[`empty_enum`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#empty_enum
[`empty_loop`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#empty_loop
[`enum_clike_unportable_variant`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#enum_clike_unportable_variant
[`enum_glob_use`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#enum_glob_use
[`enum_variant_names`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#enum_variant_names
[`eq_op`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#eq_op
[`eval_order_dependence`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#eval_order_dependence
[`expl_impl_clone_on_copy`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#expl_impl_clone_on_copy
[`explicit_counter_loop`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#explicit_counter_loop
[`explicit_into_iter_loop`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#explicit_into_iter_loop
[`explicit_iter_loop`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#explicit_iter_loop
[`extend_from_slice`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#extend_from_slice
[`filter_map`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#filter_map
[`filter_next`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#filter_next
[`float_arithmetic`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#float_arithmetic
[`float_cmp`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#float_cmp
[`for_kv_map`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#for_kv_map
[`for_loop_over_option`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#for_loop_over_option
[`for_loop_over_result`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#for_loop_over_result
[`forget_copy`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#forget_copy
[`forget_ref`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#forget_ref
[`get_unwrap`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#get_unwrap
[`identity_op`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#identity_op
[`if_let_redundant_pattern_matching`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#if_let_redundant_pattern_matching
[`if_let_some_result`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#if_let_some_result
[`if_not_else`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#if_not_else
[`if_same_then_else`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#if_same_then_else
[`ifs_same_cond`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#ifs_same_cond
[`inconsistent_digit_grouping`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#inconsistent_digit_grouping
[`indexing_slicing`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#indexing_slicing
[`ineffective_bit_mask`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#ineffective_bit_mask
[`inline_always`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#inline_always
[`integer_arithmetic`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#integer_arithmetic
[`invalid_regex`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#invalid_regex
[`invalid_upcast_comparisons`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#invalid_upcast_comparisons
[`items_after_statements`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#items_after_statements
[`iter_cloned_collect`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#iter_cloned_collect
[`iter_next_loop`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#iter_next_loop
[`iter_nth`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#iter_nth
[`iter_skip_next`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#iter_skip_next
[`iterator_step_by_zero`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#iterator_step_by_zero
[`large_digit_groups`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#large_digit_groups
[`large_enum_variant`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#large_enum_variant
[`len_without_is_empty`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#len_without_is_empty
[`len_zero`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#len_zero
[`let_and_return`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#let_and_return
[`let_unit_value`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#let_unit_value
[`linkedlist`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#linkedlist
[`logic_bug`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#logic_bug
[`manual_swap`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#manual_swap
[`many_single_char_names`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#many_single_char_names
[`map_clone`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#map_clone
[`map_entry`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#map_entry
[`match_bool`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#match_bool
[`match_overlapping_arm`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#match_overlapping_arm
[`match_ref_pats`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#match_ref_pats
[`match_same_arms`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#match_same_arms
[`match_wild_err_arm`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#match_wild_err_arm
[`mem_forget`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#mem_forget
[`min_max`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#min_max
[`misrefactored_assign_op`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#misrefactored_assign_op
[`missing_docs_in_private_items`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#missing_docs_in_private_items
[`mixed_case_hex_literals`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#mixed_case_hex_literals
[`module_inception`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#module_inception
[`modulo_one`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#modulo_one
[`mut_from_ref`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#mut_from_ref
[`mut_mut`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#mut_mut
[`mutex_atomic`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#mutex_atomic
[`mutex_integer`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#mutex_integer
[`needless_bool`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_bool
[`needless_borrow`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_borrow
[`needless_borrowed_reference`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_borrowed_reference
[`needless_continue`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_continue
[`needless_lifetimes`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_lifetimes
[`needless_pass_by_value`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_pass_by_value
[`needless_range_loop`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_range_loop
[`needless_return`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_return
[`needless_update`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#needless_update
[`neg_multiply`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#neg_multiply
[`never_loop`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#never_loop
[`new_ret_no_self`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#new_ret_no_self
[`new_without_default`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#new_without_default
[`new_without_default_derive`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#new_without_default_derive
[`no_effect`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#no_effect
[`non_ascii_literal`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#non_ascii_literal
[`nonminimal_bool`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#nonminimal_bool
[`nonsensical_open_options`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#nonsensical_open_options
[`not_unsafe_ptr_arg_deref`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#not_unsafe_ptr_arg_deref
[`ok_expect`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#ok_expect
[`op_ref`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#op_ref
[`option_map_unwrap_or`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#option_map_unwrap_or
[`option_map_unwrap_or_else`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#option_map_unwrap_or_else
[`option_unwrap_used`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#option_unwrap_used
[`or_fun_call`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#or_fun_call
[`out_of_bounds_indexing`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#out_of_bounds_indexing
[`overflow_check_conditional`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#overflow_check_conditional
[`panic_params`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#panic_params
[`partialeq_ne_impl`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#partialeq_ne_impl
[`possible_missing_comma`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#possible_missing_comma
[`precedence`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#precedence
[`print_stdout`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#print_stdout
[`print_with_newline`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#print_with_newline
[`ptr_arg`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#ptr_arg
[`pub_enum_variant_names`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#pub_enum_variant_names
[`range_step_by_zero`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#range_step_by_zero
[`range_zip_with_len`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#range_zip_with_len
[`redundant_closure`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#redundant_closure
[`redundant_closure_call`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#redundant_closure_call
[`redundant_pattern`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#redundant_pattern
[`regex_macro`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#regex_macro
[`result_unwrap_used`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#result_unwrap_used
[`reverse_range_loop`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#reverse_range_loop
[`search_is_some`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#search_is_some
[`serde_api_misuse`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#serde_api_misuse
[`shadow_reuse`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#shadow_reuse
[`shadow_same`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#shadow_same
[`shadow_unrelated`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#shadow_unrelated
[`short_circuit_statement`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#short_circuit_statement
[`should_assert_eq`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#should_assert_eq
[`should_implement_trait`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#should_implement_trait
[`similar_names`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#similar_names
[`single_char_pattern`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#single_char_pattern
[`single_match`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#single_match
[`single_match_else`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#single_match_else
[`str_to_string`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#str_to_string
[`string_add`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#string_add
[`string_add_assign`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#string_add_assign
[`string_extend_chars`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#string_extend_chars
[`string_lit_as_bytes`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#string_lit_as_bytes
[`string_to_string`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#string_to_string
[`stutter`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#stutter
[`suspicious_assignment_formatting`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#suspicious_assignment_formatting
[`suspicious_else_formatting`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#suspicious_else_formatting
[`temporary_assignment`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#temporary_assignment
[`temporary_cstring_as_ptr`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#temporary_cstring_as_ptr
[`too_many_arguments`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#too_many_arguments
[`toplevel_ref_arg`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#toplevel_ref_arg
[`transmute_ptr_to_ref`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#transmute_ptr_to_ref
[`trivial_regex`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#trivial_regex
[`type_complexity`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#type_complexity
[`unicode_not_nfc`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unicode_not_nfc
[`unit_cmp`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unit_cmp
[`unnecessary_cast`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unnecessary_cast
[`unnecessary_mut_passed`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unnecessary_mut_passed
[`unnecessary_operation`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unnecessary_operation
[`unneeded_field_pattern`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unneeded_field_pattern
[`unreadable_literal`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unreadable_literal
[`unsafe_removed_from_name`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unsafe_removed_from_name
[`unseparated_literal_suffix`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unseparated_literal_suffix
[`unstable_as_mut_slice`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unstable_as_mut_slice
[`unstable_as_slice`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unstable_as_slice
[`unused_collect`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unused_collect
[`unused_io_amount`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unused_io_amount
[`unused_label`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unused_label
[`unused_lifetimes`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#unused_lifetimes
[`use_debug`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#use_debug
[`used_underscore_binding`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#used_underscore_binding
[`useless_attribute`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_attribute
[`useless_format`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_format
[`useless_let_if_seq`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_let_if_seq
[`useless_transmute`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_transmute
[`useless_vec`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#useless_vec
[`verbose_bit_mask`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#verbose_bit_mask
[`while_let_loop`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#while_let_loop
[`while_let_on_iterator`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#while_let_on_iterator
[`wrong_pub_self_convention`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#wrong_pub_self_convention
[`wrong_self_convention`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#wrong_self_convention
[`wrong_transmute`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#wrong_transmute
[`zero_divided_by_zero`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#zero_divided_by_zero
[`zero_prefixed_literal`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#zero_prefixed_literal
[`zero_ptr`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#zero_ptr
[`zero_width_space`]: https://github.com/rust-lang-nursery/rust-clippy/wiki#zero_width_space
<!-- end autogenerated links to wiki -->
