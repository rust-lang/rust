#![feature(const_for)]
#![feature(const_mut_refs)]

const _: () = {
    for _ in 0..5 {}
    //~^ error: calls in constants are limited to
    //~| error: calls in constants are limited to
};

fn main() {}
