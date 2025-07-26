// This test exercises `-Zremap-path-scope`, diagnostics printing paths and dependency.
//
// We test different combinations with/without remap in deps, with/without remap in this
// crate but always in deps and always here but never in deps.

//@ revisions: with-diag-in-deps with-macro-in-deps with-debuginfo-in-deps
//@ revisions: only-diag-in-deps only-macro-in-deps only-debuginfo-in-deps
//@ revisions: not-diag-in-deps

//@[with-diag-in-deps] compile-flags: --remap-path-prefix={{src-base}}=remapped
//@[with-macro-in-deps] compile-flags: --remap-path-prefix={{src-base}}=remapped
//@[with-debuginfo-in-deps] compile-flags: --remap-path-prefix={{src-base}}=remapped
//@[not-diag-in-deps] compile-flags: --remap-path-prefix={{src-base}}=remapped

//@[with-diag-in-deps] compile-flags: -Zremap-path-scope=diagnostics
//@[with-macro-in-deps] compile-flags: -Zremap-path-scope=macro
//@[with-debuginfo-in-deps] compile-flags: -Zremap-path-scope=debuginfo
//@[not-diag-in-deps] compile-flags: -Zremap-path-scope=diagnostics

//@[with-diag-in-deps] aux-build:trait-diag.rs
//@[with-macro-in-deps] aux-build:trait-macro.rs
//@[with-debuginfo-in-deps] aux-build:trait-debuginfo.rs
//@[only-diag-in-deps] aux-build:trait-diag.rs
//@[only-macro-in-deps] aux-build:trait-macro.rs
//@[only-debuginfo-in-deps] aux-build:trait-debuginfo.rs
//@[not-diag-in-deps] aux-build:trait.rs

// The $SRC_DIR*.rs:LL:COL normalisation doesn't kick in automatically
// as the remapped revision will not begin with $SRC_DIR_REAL,
// so we have to do it ourselves.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:COL"

#[cfg(any(with_diag_in_deps, only_diag_in_deps))]
extern crate trait_diag as r#trait;

#[cfg(any(with_macro_in_deps, only_macro_in_deps))]
extern crate trait_macro as r#trait;

#[cfg(any(with_debuginfo_in_deps, only_debuginfo_in_deps))]
extern crate trait_debuginfo as r#trait;

#[cfg(not_diag_in_deps)]
extern crate r#trait as r#trait;

struct A;

impl r#trait::Trait for A {}
//[with-macro-in-deps]~^ ERROR `A` doesn't implement `std::fmt::Display`
//[with-debuginfo-in-deps]~^^ ERROR `A` doesn't implement `std::fmt::Display`
//[only-diag-in-deps]~^^^ ERROR `A` doesn't implement `std::fmt::Display`
//[only-macro-in-deps]~^^^^ ERROR `A` doesn't implement `std::fmt::Display`
//[only-debuginfo-in-deps]~^^^^^ ERROR `A` doesn't implement `std::fmt::Display`

//[with-diag-in-deps]~? ERROR `A` doesn't implement `std::fmt::Display`
//[not-diag-in-deps]~? ERROR `A` doesn't implement `std::fmt::Display`

fn main() {}
