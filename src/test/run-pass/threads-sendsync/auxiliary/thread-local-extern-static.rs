#![feature(cfg_target_thread_local, const_fn, thread_local)]
#![crate_type = "lib"]

#[cfg(target_thread_local)]
use std::cell::Cell;

#[no_mangle]
#[cfg(target_thread_local)]
#[thread_local]
pub static FOO: Cell<u32> = Cell::new(3);
