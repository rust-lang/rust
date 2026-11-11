//@ check-pass
#![crate_type = "lib"]
#![no_std]

extern crate std;
use ::std::*;

fn f() {
    panic!();
    //~^ WARN: `panic` is ambiguous [ambiguous_panic_imports]
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
