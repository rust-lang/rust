// Test that both `&&` and `||` actually short-circuit when the `const_if_match` feature flag is
// enabled. Without the feature flag, both sides are evaluated unconditionally.

// revisions: stock if_match

#![feature(rustc_attrs)]
#![feature(const_panic)]
#![cfg_attr(if_match, feature(const_if_match))]

const _: bool = true || panic!();  //[stock]~ ERROR any use of this value will cause an error
const _: bool = false && panic!(); //[stock]~ ERROR any use of this value will cause an error

#[rustc_error]
fn main() {} //[if_match]~ ERROR fatal error triggered by #[rustc_error]
