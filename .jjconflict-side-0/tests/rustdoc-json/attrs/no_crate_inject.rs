// Regression test for rustdoc JSON emitting
// `#[doc(no_crate_inject)]` instead of `#[doc(test(no_crate_inject))]`:
// https://github.com/rust-lang/rust/pull/153465

//@ is "$.index[?(@.inner.module.is_crate)].attrs" '[{"other": "#[doc(test(no_crate_inject))]"}]'

#![doc(test(no_crate_inject))]
