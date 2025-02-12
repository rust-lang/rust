# `rustc_private`

The tracking issue for this feature is: [#27812]

[#27812]: https://github.com/rust-lang/rust/issues/27812

------------------------

This feature allows access to unstable internal compiler crates such as `rustc_driver`.

The presence of this feature changes the way the linkage format for dylibs is calculated in a way
that is necessary for linking against dylibs that statically link `std` (such as `rustc_driver`).
This makes this feature "viral" in linkage; its use in a given crate makes its use required in
dependent crates which link to it (including integration tests, which are built as separate crates).
