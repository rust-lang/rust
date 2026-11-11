//@ aux-build:unstable_if_unmarked_const_fn_crate.rs
//@ aux-build:unmarked_const_fn_crate.rs
#![feature(staged_api, rustc_private)]
#![stable(since = "1.0.0", feature = "stable")]

extern crate unmarked_const_fn_crate;
extern crate unstable_if_unmarked_const_fn_crate;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
const fn stable_fn() {
    // This one is fine.
    unstable_if_unmarked_const_fn_crate::expose_on_stable();
    // This one is not.
    unstable_if_unmarked_const_fn_crate::not_stably_const();
    //~^ERROR: cannot use `#[feature(rustc_private)]`
    unmarked_const_fn_crate::just_a_fn();
    //~^ERROR: cannot be (indirectly) exposed to stable
}

fn main() {}
