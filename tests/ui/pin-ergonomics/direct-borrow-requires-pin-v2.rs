#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
//@ normalize-stderr: "\n\n\z" -> "\n"

struct NotPinV2;

fn direct_pin_mut(mut value: NotPinV2) {
    let _ = &pin mut value;
    //~^ ERROR cannot directly pin an ADT that is not `#[pin_v2]`
}

fn direct_pin_const(value: NotPinV2) {
    let _ = &pin const value;
    //~^ ERROR cannot directly pin an ADT that is not `#[pin_v2]`
}

fn main() {}
