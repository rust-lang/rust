// Make sure that attribute used when injecting the prelude are resolved
// hygienically.

//@ check-pass
//@ aux-build:not-libstd.rs

//@revisions: rust2015 rust2018
//@[rust2018] edition:2018

// The prelude import shouldn't see these as candidates for when it's trying to
// use the built-in macros.
extern crate core;
use core::prelude::v1::test as prelude_import;
use core::prelude::v1::test as macro_use;

// Should not be used for the prelude import - not a concern in the 2015 edition
// because `std` is already declared in the crate root.
#[cfg(rust2018)]
extern crate not_libstd as std;

#[cfg(rust2018)]
mod x {
    // The extern crate item should override `std` in the extern prelude.
    fn f() {
        std::not_in_lib_std();
    }
}

fn main() {}
