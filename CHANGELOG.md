# Change Log
All notable changes to this project will be documented in this file.

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
* Removed [`unit_expr`]
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
* New lints: [`just_underscores_and_digits`], [`result_map_unwrap_or_else`], [`transmute_bytes_to_str`]

## 0.0.168
* Rustup to *rustc 1.23.0-nightly (f0fe716db 2017-10-30)*

## 0.0.167
* Rustup to *rustc 1.23.0-nightly (90ef3372e 2017-10-29)*
* New lints: [`const_static_lifetime`], [`erasing_op`], [`fallible_impl_from`], [`println_empty_string`], [`useless_asref`]

## 0.0.166
* Rustup to *rustc 1.22.0-nightly (b7960878b 2017-10-18)*
* New lints: [`explicit_write`], [`identity_conversion`], [`implicit_hasher`], [`invalid_ref`], [`option_map_or_none`], [`range_minus_one`], [`range_plus_one`], [`transmute_int_to_bool`], [`transmute_int_to_char`], [`transmute_int_to_float`]

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
* New lint: [`unit_expr`]

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
* Add the `CLIPPY_DISABLE_DOCS_LINKS` environment variable to deactivate the
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
[`absurd_extreme_comparisons`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#absurd_extreme_comparisons
[`almost_swapped`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#almost_swapped
[`approx_constant`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#approx_constant
[`assign_op_pattern`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#assign_op_pattern
[`assign_ops`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#assign_ops
[`bad_bit_mask`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#bad_bit_mask
[`blacklisted_name`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#blacklisted_name
[`block_in_if_condition_expr`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#block_in_if_condition_expr
[`block_in_if_condition_stmt`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#block_in_if_condition_stmt
[`bool_comparison`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#bool_comparison
[`borrowed_box`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#borrowed_box
[`box_vec`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#box_vec
[`boxed_local`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#boxed_local
[`builtin_type_shadow`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#builtin_type_shadow
[`cast_lossless`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cast_lossless
[`cast_possible_truncation`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cast_possible_truncation
[`cast_possible_wrap`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cast_possible_wrap
[`cast_precision_loss`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cast_precision_loss
[`cast_ptr_alignment`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cast_ptr_alignment
[`cast_sign_loss`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cast_sign_loss
[`char_lit_as_u8`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#char_lit_as_u8
[`chars_last_cmp`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#chars_last_cmp
[`chars_next_cmp`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#chars_next_cmp
[`clone_double_ref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#clone_double_ref
[`clone_on_copy`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#clone_on_copy
[`clone_on_ref_ptr`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#clone_on_ref_ptr
[`cmp_nan`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cmp_nan
[`cmp_null`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cmp_null
[`cmp_owned`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cmp_owned
[`collapsible_if`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#collapsible_if
[`const_static_lifetime`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#const_static_lifetime
[`crosspointer_transmute`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#crosspointer_transmute
[`cyclomatic_complexity`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#cyclomatic_complexity
[`decimal_literal_representation`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#decimal_literal_representation
[`deprecated_semver`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#deprecated_semver
[`deref_addrof`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#deref_addrof
[`derive_hash_xor_eq`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#derive_hash_xor_eq
[`diverging_sub_expression`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#diverging_sub_expression
[`doc_markdown`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#doc_markdown
[`double_comparisons`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#double_comparisons
[`double_neg`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#double_neg
[`double_parens`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#double_parens
[`drop_copy`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#drop_copy
[`drop_ref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#drop_ref
[`duplicate_underscore_argument`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#duplicate_underscore_argument
[`else_if_without_else`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#else_if_without_else
[`empty_enum`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#empty_enum
[`empty_line_after_outer_attr`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#empty_line_after_outer_attr
[`empty_loop`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#empty_loop
[`enum_clike_unportable_variant`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#enum_clike_unportable_variant
[`enum_glob_use`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#enum_glob_use
[`enum_variant_names`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#enum_variant_names
[`eq_op`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#eq_op
[`erasing_op`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#erasing_op
[`eval_order_dependence`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#eval_order_dependence
[`excessive_precision`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#excessive_precision
[`expl_impl_clone_on_copy`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#expl_impl_clone_on_copy
[`explicit_counter_loop`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#explicit_counter_loop
[`explicit_into_iter_loop`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#explicit_into_iter_loop
[`explicit_iter_loop`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#explicit_iter_loop
[`explicit_write`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#explicit_write
[`extend_from_slice`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#extend_from_slice
[`fallible_impl_from`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#fallible_impl_from
[`filter_map`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#filter_map
[`filter_next`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#filter_next
[`float_arithmetic`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#float_arithmetic
[`float_cmp`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#float_cmp
[`float_cmp_const`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#float_cmp_const
[`for_kv_map`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#for_kv_map
[`for_loop_over_option`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#for_loop_over_option
[`for_loop_over_result`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#for_loop_over_result
[`forget_copy`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#forget_copy
[`forget_ref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#forget_ref
[`get_unwrap`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#get_unwrap
[`identity_conversion`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#identity_conversion
[`identity_op`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#identity_op
[`if_let_redundant_pattern_matching`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#if_let_redundant_pattern_matching
[`if_let_some_result`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#if_let_some_result
[`if_not_else`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#if_not_else
[`if_same_then_else`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#if_same_then_else
[`ifs_same_cond`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#ifs_same_cond
[`implicit_hasher`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#implicit_hasher
[`inconsistent_digit_grouping`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#inconsistent_digit_grouping
[`indexing_slicing`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#indexing_slicing
[`ineffective_bit_mask`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#ineffective_bit_mask
[`infallible_destructuring_match`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#infallible_destructuring_match
[`infinite_iter`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#infinite_iter
[`inline_always`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#inline_always
[`inline_fn_without_body`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#inline_fn_without_body
[`int_plus_one`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#int_plus_one
[`integer_arithmetic`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#integer_arithmetic
[`invalid_ref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#invalid_ref
[`invalid_regex`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#invalid_regex
[`invalid_upcast_comparisons`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#invalid_upcast_comparisons
[`items_after_statements`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#items_after_statements
[`iter_cloned_collect`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#iter_cloned_collect
[`iter_next_loop`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#iter_next_loop
[`iter_nth`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#iter_nth
[`iter_skip_next`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#iter_skip_next
[`iterator_step_by_zero`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#iterator_step_by_zero
[`just_underscores_and_digits`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#just_underscores_and_digits
[`large_digit_groups`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#large_digit_groups
[`large_enum_variant`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#large_enum_variant
[`len_without_is_empty`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#len_without_is_empty
[`len_zero`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#len_zero
[`let_and_return`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#let_and_return
[`let_unit_value`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#let_unit_value
[`linkedlist`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#linkedlist
[`logic_bug`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#logic_bug
[`manual_memcpy`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#manual_memcpy
[`manual_swap`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#manual_swap
[`many_single_char_names`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#many_single_char_names
[`map_clone`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#map_clone
[`map_entry`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#map_entry
[`match_as_ref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#match_as_ref
[`match_bool`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#match_bool
[`match_overlapping_arm`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#match_overlapping_arm
[`match_ref_pats`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#match_ref_pats
[`match_same_arms`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#match_same_arms
[`match_wild_err_arm`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#match_wild_err_arm
[`maybe_infinite_iter`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#maybe_infinite_iter
[`mem_forget`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#mem_forget
[`min_max`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#min_max
[`misaligned_transmute`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#misaligned_transmute
[`misrefactored_assign_op`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#misrefactored_assign_op
[`missing_docs_in_private_items`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#missing_docs_in_private_items
[`mixed_case_hex_literals`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#mixed_case_hex_literals
[`module_inception`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#module_inception
[`modulo_one`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#modulo_one
[`mut_from_ref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#mut_from_ref
[`mut_mut`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#mut_mut
[`mut_range_bound`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#mut_range_bound
[`mutex_atomic`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#mutex_atomic
[`mutex_integer`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#mutex_integer
[`naive_bytecount`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#naive_bytecount
[`needless_bool`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#needless_bool
[`needless_borrow`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#needless_borrow
[`needless_borrowed_reference`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#needless_borrowed_reference
[`needless_continue`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#needless_continue
[`needless_lifetimes`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#needless_lifetimes
[`needless_pass_by_value`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#needless_pass_by_value
[`needless_range_loop`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#needless_range_loop
[`needless_return`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#needless_return
[`needless_update`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#needless_update
[`neg_multiply`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#neg_multiply
[`never_loop`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#never_loop
[`new_ret_no_self`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#new_ret_no_self
[`new_without_default`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#new_without_default
[`new_without_default_derive`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#new_without_default_derive
[`no_effect`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#no_effect
[`non_ascii_literal`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#non_ascii_literal
[`nonminimal_bool`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#nonminimal_bool
[`nonsensical_open_options`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#nonsensical_open_options
[`not_unsafe_ptr_arg_deref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#not_unsafe_ptr_arg_deref
[`ok_expect`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#ok_expect
[`op_ref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#op_ref
[`option_map_or_none`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#option_map_or_none
[`option_map_unit_fn`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#option_map_unit_fn
[`option_map_unwrap_or`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#option_map_unwrap_or
[`option_map_unwrap_or_else`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#option_map_unwrap_or_else
[`option_option`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#option_option
[`option_unwrap_used`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#option_unwrap_used
[`or_fun_call`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#or_fun_call
[`out_of_bounds_indexing`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#out_of_bounds_indexing
[`overflow_check_conditional`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#overflow_check_conditional
[`panic_params`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#panic_params
[`partialeq_ne_impl`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#partialeq_ne_impl
[`possible_missing_comma`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#possible_missing_comma
[`precedence`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#precedence
[`print_literal`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#print_literal
[`print_stdout`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#print_stdout
[`print_with_newline`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#print_with_newline
[`println_empty_string`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#println_empty_string
[`ptr_arg`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#ptr_arg
[`pub_enum_variant_names`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#pub_enum_variant_names
[`question_mark`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#question_mark
[`range_minus_one`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#range_minus_one
[`range_plus_one`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#range_plus_one
[`range_step_by_zero`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#range_step_by_zero
[`range_zip_with_len`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#range_zip_with_len
[`redundant_closure`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#redundant_closure
[`redundant_closure_call`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#redundant_closure_call
[`redundant_field_names`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#redundant_field_names
[`redundant_pattern`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#redundant_pattern
[`regex_macro`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#regex_macro
[`replace_consts`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#replace_consts
[`result_map_unit_fn`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#result_map_unit_fn
[`result_map_unwrap_or_else`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#result_map_unwrap_or_else
[`result_unwrap_used`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#result_unwrap_used
[`reverse_range_loop`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#reverse_range_loop
[`search_is_some`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#search_is_some
[`serde_api_misuse`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#serde_api_misuse
[`shadow_reuse`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#shadow_reuse
[`shadow_same`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#shadow_same
[`shadow_unrelated`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#shadow_unrelated
[`short_circuit_statement`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#short_circuit_statement
[`should_assert_eq`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#should_assert_eq
[`should_implement_trait`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#should_implement_trait
[`similar_names`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#similar_names
[`single_char_pattern`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#single_char_pattern
[`single_match`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#single_match
[`single_match_else`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#single_match_else
[`str_to_string`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#str_to_string
[`string_add`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#string_add
[`string_add_assign`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#string_add_assign
[`string_extend_chars`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#string_extend_chars
[`string_lit_as_bytes`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#string_lit_as_bytes
[`string_to_string`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#string_to_string
[`stutter`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#stutter
[`suspicious_arithmetic_impl`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#suspicious_arithmetic_impl
[`suspicious_assignment_formatting`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#suspicious_assignment_formatting
[`suspicious_else_formatting`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#suspicious_else_formatting
[`suspicious_op_assign_impl`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#suspicious_op_assign_impl
[`temporary_assignment`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#temporary_assignment
[`temporary_cstring_as_ptr`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#temporary_cstring_as_ptr
[`too_many_arguments`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#too_many_arguments
[`toplevel_ref_arg`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#toplevel_ref_arg
[`transmute_bytes_to_str`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#transmute_bytes_to_str
[`transmute_int_to_bool`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#transmute_int_to_bool
[`transmute_int_to_char`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#transmute_int_to_char
[`transmute_int_to_float`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#transmute_int_to_float
[`transmute_ptr_to_ptr`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#transmute_ptr_to_ptr
[`transmute_ptr_to_ref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#transmute_ptr_to_ref
[`trivial_regex`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#trivial_regex
[`type_complexity`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#type_complexity
[`unicode_not_nfc`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unicode_not_nfc
[`unit_arg`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unit_arg
[`unit_cmp`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unit_cmp
[`unnecessary_cast`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unnecessary_cast
[`unnecessary_fold`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unnecessary_fold
[`unnecessary_mut_passed`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unnecessary_mut_passed
[`unnecessary_operation`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unnecessary_operation
[`unneeded_field_pattern`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unneeded_field_pattern
[`unreadable_literal`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unreadable_literal
[`unsafe_removed_from_name`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unsafe_removed_from_name
[`unseparated_literal_suffix`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unseparated_literal_suffix
[`unstable_as_mut_slice`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unstable_as_mut_slice
[`unstable_as_slice`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unstable_as_slice
[`unused_collect`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unused_collect
[`unused_io_amount`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unused_io_amount
[`unused_label`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unused_label
[`unused_lifetimes`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#unused_lifetimes
[`use_debug`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#use_debug
[`use_self`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#use_self
[`used_underscore_binding`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#used_underscore_binding
[`useless_asref`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#useless_asref
[`useless_attribute`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#useless_attribute
[`useless_format`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#useless_format
[`useless_let_if_seq`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#useless_let_if_seq
[`useless_transmute`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#useless_transmute
[`useless_vec`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#useless_vec
[`verbose_bit_mask`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#verbose_bit_mask
[`while_immutable_condition`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#while_immutable_condition
[`while_let_loop`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#while_let_loop
[`while_let_on_iterator`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#while_let_on_iterator
[`write_literal`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#write_literal
[`write_with_newline`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#write_with_newline
[`writeln_empty_string`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#writeln_empty_string
[`wrong_pub_self_convention`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#wrong_pub_self_convention
[`wrong_self_convention`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#wrong_self_convention
[`wrong_transmute`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#wrong_transmute
[`zero_divided_by_zero`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#zero_divided_by_zero
[`zero_prefixed_literal`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#zero_prefixed_literal
[`zero_ptr`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#zero_ptr
[`zero_width_space`]: https://rust-lang-nursery.github.io/rust-clippy/master/index.html#zero_width_space
<!-- end autogenerated links to wiki -->
