# Change Log
All notable changes to this project will be documented in this file.

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
* New lint: [`non_expressive_names`]

## 0.0.55 — 2016-03-21
* Update to *rustc 1.9.0-nightly (02310fd31 2016-03-19)*

## 0.0.54 — 2016-03-16
* Update to *rustc 1.9.0-nightly (c66d2380a 2016-03-15)*

## 0.0.53 — 2016-03-15
* Add a [configuration file]

## ~~0.0.52~~

## 0.0.51 — 2016-03-13
* Add `str` to types considered by `len_zero`
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

[configuration file]: ./rust-clippy#configuration

[`crosspointer_transmute`]: https://github.com/Manishearth/rust-clippy/wiki#crosspointer_transmute
[`doc_markdown`]: https://github.com/Manishearth/rust-clippy/wiki#doc_markdown
[`indexing_slicing`]: https://github.com/Manishearth/rust-clippy/wiki#indexing_slicing
[`invalid_upcast_comparisons`]: https://github.com/Manishearth/rust-clippy/wiki#invalid_upcast_comparisons
[`logic_bug`]: https://github.com/Manishearth/rust-clippy/wiki#logic_bug
[`match_same_arms`]: https://github.com/Manishearth/rust-clippy/wiki#match_same_arms
[`needless_range_loop`]: https://github.com/Manishearth/rust-clippy/wiki#needless_range_loop
[`new_without_default`]: https://github.com/Manishearth/rust-clippy/wiki#new_without_default
[`non_expressive_names`]: https://github.com/Manishearth/rust-clippy/wiki#non_expressive_names
[`nonminimal_bool`]: https://github.com/Manishearth/rust-clippy/wiki#nonminimal_bool
[`overflow_check_conditional`]: https://github.com/Manishearth/rust-clippy/wiki#overflow_check_conditional
[`redundant_closure_call`]: https://github.com/Manishearth/rust-clippy/wiki#redundant_closure_call
[`str_to_string`]: https://github.com/Manishearth/rust-clippy/wiki#str_to_string
[`string_to_string`]: https://github.com/Manishearth/rust-clippy/wiki#string_to_string
[`unstable_as_mut_slice`]: https://github.com/Manishearth/rust-clippy/wiki#unstable_as_mut_slice
[`unstable_as_slice`]: https://github.com/Manishearth/rust-clippy/wiki#unstable_as_slice
[`unused_label`]: https://github.com/Manishearth/rust-clippy/wiki#unused_label
[`useless_vec`]: https://github.com/Manishearth/rust-clippy/wiki#useless_vec
