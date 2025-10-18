//@ run-pass
//@ check-run-results
#![feature(eii)]

#[eii]
fn hello(x: u64);

#[hello]
fn hello_impl(x: u64) {
    println!("{x:?}")
}

// what you would write:
fn main() {
    // directly
    hello_impl(21);
    // through the alias
    hello(42);
}
