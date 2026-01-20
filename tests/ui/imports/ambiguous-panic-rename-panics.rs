//@ edition: 2024
#![crate_type = "lib"]

mod m1 {
    pub use core::prelude::v1::panic as p;
}

mod m2 {
    pub use std::prelude::v1::panic as p;
}

use m2::*;
fn xx() {
    use m1::*;

    p!(); //~ ERROR: `p` is ambiguous [E0659]
}
