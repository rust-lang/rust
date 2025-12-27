//! Ensure we reject invalid combinations of feature gating inline consts

#![feature(staged_api, abi_unadjusted)]
#![stable(feature = "rust_test", since = "1.0.0")]

#[unstable(feature = "abi_unadjusted", issue = "42")]
#[rustc_const_unstable(feature = "abi_unadjusted", issue = "42")]
const fn my_fun() {}

#[stable(feature = "asdf", since = "99.0.0")]
#[rustc_const_stable(feature = "asdf", since = "99.0.0")]
const fn my_fun2() {
    #[rustc_const_unstable(feature = "abi_unadjusted", issue = "42")]
    //~^ ERROR: must match const stability of containing item
    const {
        my_fun()
    }
}

// Check that const stable const blocks can only call const stable things
#[stable(feature = "asdf", since = "99.0.0")]
#[rustc_const_stable(feature = "asdf", since = "99.0.0")]
const fn my_fun3() {
    #[rustc_const_stable(feature = "asdf", since = "99.0.0")]
    const {
        my_fun()
        //~^ ERROR: (indirectly) exposed to stable
    }
}

// Check that const stable const blocks can only call const stable things
#[stable(feature = "asdf", since = "99.0.0")]
#[rustc_const_stable(feature = "asdf", since = "99.0.0")]
const fn my_fun4() {
    const {
        my_fun()
        //~^ ERROR: (indirectly) exposed to stable
    }
}

fn main() {
    #[rustc_const_unstable(feature = "abi_unadjusted", issue = "42")]
    //~^ ERROR: must match const stability of containing item
    const {
        my_fun2()
    };
}
