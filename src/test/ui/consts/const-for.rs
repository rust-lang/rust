#![feature(const_for)]
#![feature(const_mut_refs)]

const _: () = {
    for _ in 0..5 {}
    //~^ error: cannot convert
    //~| error: cannot call non-const fn
};

fn main() {}
