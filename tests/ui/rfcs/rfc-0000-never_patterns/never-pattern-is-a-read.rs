// Make sure we consider `!` to be a union read.

#![feature(never_type, never_patterns)]

union U {
    a: !,
    b: usize,
}

fn foo<T>(u: U) -> ! {
    let U { a: ! } = u;
    //~^ ERROR access to union field is unsafe
}

fn main() {}
