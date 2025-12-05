//@ revisions: allow not_allow
//@ compile-flags: --crate-type=lib -Cinstrument-coverage  -Zno-profiler-runtime
//@[allow] check-pass

#![feature(staged_api, rustc_allow_const_fn_unstable)]
#![stable(feature = "rust_test", since = "1.0.0")]

#[stable(feature = "rust_test", since = "1.0.0")]
#[rustc_const_stable(feature = "rust_test", since = "1.0.0")]
#[cfg_attr(allow, rustc_allow_const_fn_unstable(const_precise_live_drops))]
pub const fn unwrap<T>(this: Option<T>) -> T {
//[not_allow]~^ ERROR: cannot be evaluated
    match this {
        Some(x) => x,
        None => panic!(),
    }
}
