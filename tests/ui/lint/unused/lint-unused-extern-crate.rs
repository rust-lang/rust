//@ aux-build:lint_unused_extern_crate.rs
//@ aux-build:lint_unused_extern_crate2.rs
//@ aux-build:lint_unused_extern_crate3.rs
//@ aux-build:lint_unused_extern_crate4.rs
//@ aux-build:lint_unused_extern_crate5.rs

#![deny(unused_extern_crates)]
#![allow(unused_variables)]
#![allow(deprecated)]

extern crate lint_unused_extern_crate5; //~ ERROR: unused extern crate

pub extern crate lint_unused_extern_crate4; // no error, it is re-exported

extern crate lint_unused_extern_crate3; // no error, it is used

extern crate lint_unused_extern_crate2; // no error, the use marks it as used
                                        // even if imported objects aren't used

extern crate lint_unused_extern_crate as other; // no error, the use * marks it as used

#[allow(unused_imports)]
use lint_unused_extern_crate2::foo as bar;

use other::*;

mod foo {
    // Test that this is unused even though an earlier `extern crate` is used.
    extern crate lint_unused_extern_crate2; //~ ERROR unused extern crate
}

fn main() {
    lint_unused_extern_crate3::foo();
    foo();
}
