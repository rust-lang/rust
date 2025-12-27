//! Ensure that we can use a language feature with a `const fn`:
//! enabling the feature gate actually lets us call the function.
//@ check-pass

#![feature(staged_api, abi_unadjusted, rustc_allow_const_fn_unstable)]
#![stable(feature = "rust_test", since = "1.0.0")]

#[unstable(feature = "abi_unadjusted", issue = "42")]
#[rustc_const_unstable(feature = "abi_unadjusted", issue = "42")]
const fn my_fun() {}

#[unstable(feature = "abi_unadjusted", issue = "42")]
#[rustc_const_unstable(feature = "abi_unadjusted", issue = "42")]
const fn my_fun2() {
    my_fun()
}

// Check that we can call unstable things in unstable const blocks
// in unstable fns.
#[unstable(feature = "abi_unadjusted", issue = "42")]
#[rustc_const_unstable(feature = "abi_unadjusted", issue = "42")]
const fn my_fun3() {
    #[rustc_const_unstable(feature = "abi_unadjusted", issue = "42")]
    const {
        my_fun()
    }
}

#[stable(feature = "asdf", since = "99.0.0")]
#[rustc_const_stable(feature = "asdf", since = "99.0.0")]
const fn stable_thing() {}

#[stable(feature = "asdf", since = "99.0.0")]
#[rustc_const_stable(feature = "asdf", since = "99.0.0")]
const fn my_fun4() {
    #[rustc_const_stable(feature = "asdf", since = "99.0.0")]
    const {
        stable_thing()
    }
}

#[stable(feature = "asdf", since = "99.0.0")]
#[rustc_const_stable(feature = "asdf", since = "99.0.0")]
const fn my_fun5() {
    const { stable_thing() }
}

fn main() {
    #[rustc_allow_const_fn_unstable(abi_unadjusted)]
    const {
        my_fun2()
    };
}
