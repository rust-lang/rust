#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

const fn f() {
    become dangerous();
    //~^ error: call to unsafe function `dangerous` is unsafe and requires unsafe function or block
}

const unsafe fn dangerous() {}

fn main() {}
