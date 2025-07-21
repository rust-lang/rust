//@ edition:2018
//@ aux-build: remove-extern-crate.rs
//@ run-rustfix

#![warn(rust_2018_idioms)]

#[cfg_attr(test, "macro_use")] //~ ERROR expected
extern crate remove_extern_crate as foo; //~ WARNING unused extern crate
extern crate core; //~ WARNING unused extern crate

mod another {
    #[cfg_attr(test)] //~ ERROR expected
    extern crate remove_extern_crate as foo; //~ WARNING unused extern crate
    extern crate core; //~ WARNING unused extern crate
}

fn main() {}
