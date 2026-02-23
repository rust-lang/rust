//@ edition: 2018
//@ check-pass
#![crate_type = "lib"]
use ::core::prelude::v1::*;

fn f() {
    panic!(&std::string::String::new());
    //~^ WARN: `panic` is ambiguous [ambiguous_panic_imports]
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| WARN: panic message is not a string literal [non_fmt_panics]
}
