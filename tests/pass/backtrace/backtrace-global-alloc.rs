//@compile-flags: -Zmiri-disable-isolation

#![feature(backtrace)]

use std::alloc::System;
use std::backtrace::Backtrace;

#[global_allocator]
static GLOBAL_ALLOCATOR: System = System;

fn main() {
    eprint!("{}", Backtrace::capture());
}
