//@ compile-flags: --crate-type=lib

#![feature(unsized_fn_params)]

pub fn f(k: dyn std::fmt::Display) {
    let k2 = move || {
        k.to_string();
        //~^ ERROR the size for values of type `(dyn std::fmt::Display + 'static)` cannot be known at compilation time
    };
}
