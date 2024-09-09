# `rustc_private`

The tracking issue for this feature is: [#27812]

[#27812]: https://github.com/rust-lang/rust/issues/27812

------------------------

This feature allows access to unstable internal compiler crates.

Additionally it changes the linking behavior of crates which have this feature enabled. It will prevent linking to a dylib if there's a static variant of it already statically linked into another dylib dependency. This is required to successfully link to `rustc_driver`.
