//@compile-flags: -Zmiri-disable-isolation
//@rustc-env: RUST_BACKTRACE=1

use std::alloc::System;
use std::backtrace::Backtrace;

#[global_allocator]
static GLOBAL_ALLOCATOR: System = System;

fn main() {
    eprint!("{}", Backtrace::capture());
}
