// ignore-tidy-linelength
// ignore-compare-mode-nll
#![feature(const_in_array_repeat_expressions, nll)]
#![allow(warnings)]

// Some type that is not copyable.
struct Bar;

const fn type_no_copy() -> Option<Bar> {
    None
}

const fn type_copy() -> u32 {
    3
}

fn no_copy() {
    const ARR: [Option<Bar>; 2] = [type_no_copy(); 2];
    //~^ ERROR the trait bound `std::option::Option<Bar>: std::marker::Copy` is not satisfied [E0277]
}

fn copy() {
    const ARR: [u32; 2] = [type_copy(); 2];
}

fn main() {}
