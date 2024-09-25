#![feature(const_for)]

const _: () = {
    for _ in 0..5 {}
    //~^ error: cannot call
    //~| error: cannot convert
};

fn main() {}
