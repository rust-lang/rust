//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024

use std::env;

#[deny(unsafe_op_in_unsafe_fn)]
unsafe fn unsafe_fn() {
    env::set_var("FOO", "BAR");
    //[e2024]~^ ERROR call to unsafe function `std::env::set_var` is unsafe
    env::remove_var("FOO");
    //[e2024]~^ ERROR call to unsafe function `std::env::remove_var` is unsafe
    if false {
        unsafe_fn();
        //~^ ERROR call to unsafe function `unsafe_fn` is unsafe
    }
}
fn safe_fn() {}

#[deny(unused_unsafe)]
fn main() {
    env::set_var("FOO", "BAR");
    //[e2024]~^ ERROR call to unsafe function `set_var` is unsafe
    env::remove_var("FOO");
    //[e2024]~^ ERROR call to unsafe function `remove_var` is unsafe

    unsafe {
        env::set_var("FOO", "BAR");
        env::remove_var("FOO");
    }

    unsafe_fn();
    //~^ ERROR call to unsafe function `unsafe_fn` is unsafe

    unsafe {
        //~^ ERROR unnecessary `unsafe` block
        safe_fn();
    }
}
