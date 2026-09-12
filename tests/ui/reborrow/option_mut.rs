//! Test that Option<&mut ()> can be reborrowed.
//! This should pass eventually.

#![feature(reborrow)]

fn method(a: Option<&mut ()>) {}

fn main() {
    let a = Some(&mut ());
    let _ = method(a);
    let _ = method(a); //~ERROR use of moved value: `a`
}
