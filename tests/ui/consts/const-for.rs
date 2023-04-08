#![feature(const_for)]
#![feature(const_mut_refs)]

const _: () = {
    for _ in 0..5 {}
    //~^ error: cannot call
    //~| error: cannot convert
    //~| error: the trait bound
};

fn main() {}
