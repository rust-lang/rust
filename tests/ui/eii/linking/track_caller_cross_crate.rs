//@ run-pass
//@ check-run-results
//@ aux-build: track_caller_cross_other_crate.rs
//@ compile-flags: -O
//@ ignore-backends: gcc
//@ ignore-windows
// Tests that `#[track_caller]` on an EII declaration in one crate is derived
// onto an explicit implementation in another crate.

extern crate track_caller_cross_other_crate as decl;

use std::panic::Location;
use std::sync::atomic::{AtomicU32, Ordering};

static LAST_LINE: AtomicU32 = AtomicU32::new(0);

#[decl::tcross]
fn my_impl(x: u64) {
    LAST_LINE.store(Location::caller().line(), Ordering::SeqCst);
    println!("impl {x}");
}

fn main() {
    my_impl(1);
    assert_eq!(LAST_LINE.load(Ordering::SeqCst), line!() - 1);

    decl::tcross(2);
    assert_eq!(LAST_LINE.load(Ordering::SeqCst), line!() - 1);
}
