//@ aux-build:on-type-error.rs
extern crate on_type_error;

use on_type_error::S;

struct K<T> {
    foo: T,
}
fn main() {
    let s: S<i32> = S(String::new());
    //~^ ERROR mismatched types
    //~| NOTE arguments to this struct are incorrect
    //~| NOTE expected `i32`, found `String`
    //~| NOTE this argument influences the type of `S`
    //~| NOTE tuple struct defined here
    let k: K<i32> = K { foo: "" };
    //~^ ERROR mismatched types
    //~| NOTE expected `i32`, found `&str`
    let _: S<i32> = k;
    //~^ ERROR mismatched types
    //~| NOTE expected due to this
    //~| NOTE expected `S<i32>`, found `K<i32>`
    //~| NOTE custom on_type_error note: expected struct `S<i32>`
    //~| NOTE expected struct `S<i32>`
}
