#![feature(custom_inner_attributes)]
#![clippy::msrv = "invalid.version"]
//~^ ERROR: `invalid.version` is not a valid Rust version

fn main() {}

#[clippy::msrv = "invalid.version"]
//~^ ERROR: `invalid.version` is not a valid Rust version
fn outer_attr() {}

mod multiple {
    #![clippy::msrv = "1.40"]
    #![clippy::msrv = "=1.35.0"]
    //~^ ERROR: `msrv` is defined multiple times
    #![clippy::msrv = "1.10.1"]
    //~^ ERROR: `msrv` is defined multiple times

    mod foo {
        #![clippy::msrv = "1"]
        #![clippy::msrv = "1.0.0"]
        //~^ ERROR: `msrv` is defined multiple times
    }
}
