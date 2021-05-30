#![crate_type = "lib"]
#![feature(thread_local_const_init)]

use std::cell::Cell;

thread_local!(pub static A: Cell<u64> = const { Cell::new(0) });
