//@ edition:2015
//@ check-pass

#![feature(async_for_loop)]

// Make sure we don't break `for await` loops in the 2015 edition, where `await` was allowed as an
// identifier.

fn main() {
    for await in 0..3 {}
}
