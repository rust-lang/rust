#![crate_type = "lib"]

use std::cell::Cell;

thread_local!(pub static A: Cell<u64> = const { Cell::new(0) });
