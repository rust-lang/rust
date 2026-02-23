#![deny(ambiguous_panic_imports)]
#![crate_type = "lib"]
#![no_std]

extern crate std;
use std::prelude::v1::*;

fn xx() {
    panic!();
    //~^ ERROR `panic` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
