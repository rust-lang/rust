//@ edition: 2024
//@ check-pass
#![crate_type = "lib"]
mod m1 {
    pub use core::prelude::v1::*;
}

mod m2 {
    pub use std::prelude::v1::*;
}

#[allow(unused)]
use m2::*;
fn foo() {
    use m1::*;

    panic!();
    //~^ WARN: `panic` is ambiguous [ambiguous_panic_imports]
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
