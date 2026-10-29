//@run-rustfix
#![allow(dead_code)]

trait Trait {}

fn assert_send() -> *mut dyn (Trait + Send) {
    //~^ ERROR incorrect parentheses around trait bounds
    loop {}
}

fn foo2(_: &dyn (Trait + Send)) {}
//~^ ERROR incorrect parentheses around trait bounds

fn foo3(_: &dyn(Trait + Send)) {}
//~^ ERROR incorrect parentheses around trait bounds

fn main() {}
