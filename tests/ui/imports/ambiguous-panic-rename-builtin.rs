//@ edition: 2024
#![crate_type = "lib"]
#![no_std]

extern crate std;
mod m1 {
    pub use std::prelude::v1::env as panic;
}
use m1::*;

fn xx() {
    panic!();
    //~^ ERROR: `env!()` takes 1 or 2 arguments
    //~| ERROR: `panic` is ambiguous [E0659]
}
