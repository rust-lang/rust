//@ edition: 2024
#![crate_type = "lib"]
#![no_implicit_prelude]

mod m1 {
    macro_rules! panic {
        () => {};
    }

    pub(crate) use panic;
}

fn foo() {
    use m1::*;
    panic!(); //~ERROR: `panic` is ambiguous [E0659]
}
