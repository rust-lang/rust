//@ build-pass
//@ compile-flags: --crate-type=lib

#![feature(unsized_fn_params)]

pub fn f(k: dyn std::fmt::Display) {
    let k2 = || {
        k.to_string();
    };
}
