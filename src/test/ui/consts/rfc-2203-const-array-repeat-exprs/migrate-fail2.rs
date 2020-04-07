// ignore-compare-mode-nll
// compile-flags: -Z borrowck=migrate
// build-fail
#![feature(const_in_array_repeat_expressions, const_panic, const_fn)]
#![allow(warnings)]

// Some type that is not copyable.
struct Bar;

const fn bad_bar() -> Bar { panic!() }
//~^ ERROR evaluation of constant value failed

fn main() {
    let arr: [Bar; 2] = [bad_bar(); 2];
    //~^ ERROR erroneous constant used
    //~| ERROR evaluation of constant expression failed
}
