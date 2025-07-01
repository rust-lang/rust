//@ aux-build:unexported-type-error-message.rs

extern crate unexported_type_error_message;

fn main() {
    // Here, the type returned by foo() is not exported.
    // This used to cause internal errors when serializing
    // because the def_id associated with the type was
    // not convertible to a path.
    let x: isize = unexported_type_error_message::foo();
    //~^ ERROR mismatched types
    //~| NOTE expected type `isize`
    //~| NOTE found enum `Option<isize>`
    //~| NOTE expected `isize`, found `Option<isize>`
    //~| NOTE expected due to this
}
