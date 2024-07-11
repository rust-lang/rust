// Check that we correctly render late-bound lifetime params in source order
// even if early-bound generic params are present.
//
// For context, at the time of writing early- and late-bound params are stored
// separately in rustc and therefore rustdoc needs to manually merge them.

#![crate_name = "usr"]
//@ aux-crate:dep=early-late-bound-lifetime-params.rs
//@ edition:2021

//@ has usr/fn.f.html
//@ has - '//pre[@class="rust item-decl"]' "fn f<'a, 'b, 'c, 'd, T, const N: usize>(_: impl Copy)"
pub use dep::f;

//@ has usr/struct.Ty.html
//@ has - '//*[@id="method.f"]' "fn f<'a, 'b, 'c, 'd, T, const N: usize>(_: impl Copy)"
pub use dep::Ty;
