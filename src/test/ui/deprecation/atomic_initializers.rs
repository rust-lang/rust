// run-rustfix
// build-pass (FIXME(62277): could be check-pass?)

#[allow(deprecated, unused_imports)]
use std::sync::atomic::{AtomicIsize, ATOMIC_ISIZE_INIT};

#[allow(dead_code)]
static FOO: AtomicIsize = ATOMIC_ISIZE_INIT;
//~^ WARN use of deprecated item

fn main() {}
