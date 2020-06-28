#![feature(const_generics)]
#![allow(incomplete_features)]

fn func<A, const F: fn(inner: A)>(outer: A) {
    //~^ ERROR: using function pointers as const generic parameters is forbidden
    F(outer);
}

fn main() {}
