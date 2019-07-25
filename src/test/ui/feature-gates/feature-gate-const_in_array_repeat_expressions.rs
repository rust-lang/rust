// ignore-tidy-linelength
#![allow(warnings)]

struct Bar;

fn foo() {
    let arr: [Option<String>; 2] = [None::<String>; 2];
    //~^ ERROR the trait bound `std::option::Option<std::string::String>: std::marker::Copy` is not satisfied [E0277]
}

fn main() {}
