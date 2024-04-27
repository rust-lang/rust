//@ run-rustfix
//@ check-pass

#[allow(deprecated, unused_imports)]
use std::sync::atomic::{AtomicIsize, ATOMIC_ISIZE_INIT};

#[allow(dead_code)]
static FOO: AtomicIsize = ATOMIC_ISIZE_INIT;
//~^ WARN use of deprecated constant

fn main() {}
