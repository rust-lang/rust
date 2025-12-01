//@ edition: 2018
#![crate_type = "lib"]
use ::core::prelude::v1::*;

#[allow(unused)]
fn f() {
    panic!(std::string::String::new());
    //~^ WARN: `panic` is ambiguous [ambiguous_panic_imports]
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| ERROR: mismatched types [E0308]
}
