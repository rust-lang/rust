// gate-test-const_fn_trait_bound

// revisions: stock gated

#![feature(rustc_attrs)]
#![cfg_attr(gated, feature(const_fn_trait_bound))]

const fn test1<T: std::ops::Add>() {}
//[stock]~^ trait bounds
const fn test2(_x: &dyn Send) {}
//[stock]~^ trait bounds
const fn test3() -> &'static dyn Send { loop {} }
//[stock]~^ trait bounds


#[rustc_error]
fn main() {} //[gated]~ fatal error triggered by #[rustc_error]
