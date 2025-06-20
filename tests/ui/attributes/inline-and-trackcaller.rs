//! Regression test for <https://github.com/rust-lang/rust/issues/142783>.

#![crate_type = "lib"]

#[inline] //~ERROR attribute should be applied to function or closure
#[track_caller]
macro_rules! contained {
    () => {};
}
