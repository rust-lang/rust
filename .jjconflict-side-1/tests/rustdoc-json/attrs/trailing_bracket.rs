// Regression test for rustdoc JSON emitting
// `#[doc(test(attr(deny(rust_2018_idioms))))` without the trailing `]`:
// https://github.com/rust-lang/rust/pull/153465

//@ is "$.index[?(@.inner.module.is_crate)].attrs" '[{"other": "#[doc(test(attr(deny(rust_2018_idioms))))]"}]'

#![doc(test(attr(deny(rust_2018_idioms))))]
