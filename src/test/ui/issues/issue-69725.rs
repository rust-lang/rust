// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

// aux-build:issue-69725.rs

extern crate issue_69725;
use issue_69725::Struct;

fn crash<A>() {
    let _ = Struct::<A>::new().clone();
    //~^ ERROR: no method named `clone` found
}

fn main() {}
