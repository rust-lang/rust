//! Ensure that we can use a language feature with a `const fn`:
//! enabling the feature gate actually lets us call the function.
//@ check-pass

#![feature(staged_api, abi_unadjusted)]
#![stable(feature = "rust_test", since = "1.0.0")]

#[unstable(feature = "abi_unadjusted", issue = "42")]
#[rustc_const_unstable(feature = "abi_unadjusted", issue = "42")]
const fn my_fun() {}

#[unstable(feature = "abi_unadjusted", issue = "42")]
#[rustc_const_unstable(feature = "abi_unadjusted", issue = "42")]
const fn my_fun2() {
    my_fun()
}

fn main() {
    const { my_fun2() };
}
