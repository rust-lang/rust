// ignore-tidy-linelength
#![allow(warnings)]

struct Bar;

// This function would compile with the feature gate, and tests that it is suggested.
fn foo() {
    let arr: [Option<String>; 2] = [None::<String>; 2];
    //~^ ERROR the trait bound `std::option::Option<std::string::String>: std::marker::Copy` is not satisfied [E0277]
}

// This function would not compile with the feature gate, and tests that it is not suggested.
fn bar() {
    let arr: [Option<String>; 2] = [Some("foo".to_string()); 2];
    //~^ ERROR the trait bound `std::option::Option<std::string::String>: std::marker::Copy` is not satisfied [E0277]
}

fn main() {}
