//@ check-fail

#![feature(non_lifetime_binders)]

fn take() -> impl for<T> Fn() -> T {
    //~^ ERROR type mismatch
    || 3
}

fn main() {}
