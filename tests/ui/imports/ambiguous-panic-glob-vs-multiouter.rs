//@ edition: 2024
#![crate_type = "lib"]
mod m1 {
    pub use core::prelude::v1::*;
}

mod m2 {
    pub use std::prelude::v1::*;
}

use m2::*;
fn foo() {
    use m1::*;

    panic!(); //~ ERROR: `panic` is ambiguous [E0659]
}
