// Make sure we consider `!` to be a union read.

#![feature(never_type, never_patterns)]
//~^ WARN the feature `never_patterns` is incomplete

union U {
    a: !,
    b: usize,
}

fn foo<T>(u: U) -> ! {
    let U { a: ! } = u;
    //~^ ERROR access to union field is unsafe
}

fn main() {}
