//@ edition: 2024
#![crate_type = "lib"]
mod m1 {
    pub use core::prelude::v1::*;
}

mod m2 {
    pub use std::prelude::v1::*;
}

fn foo() {
    use m1::*;
    use m2::*;

    // I had hoped that this would not produce the globvsglob error because it would never be
    // resolving `panic` via one of the ambiguous glob imports above but it appears to do so, not
    // sure why
    panic!();
    //~^ WARN: `panic` is ambiguous [ambiguous_panic_imports]
    //~| WARN: this was previously accepted by the compiler
    //~| ERROR: `panic` is ambiguous [ambiguous_glob_imports]
    //~| WARN: this was previously accepted by the compiler
}
