#![crate_type = "dylib"]
#![feature(thread_local)]
#![feature(cfg_target_thread_local)]

extern crate tls_rlib;

pub use tls_rlib::*;

#[cfg(target_thread_local)]
#[thread_local]
pub static FOO: bool = true;

#[cfg(target_thread_local)]
#[inline(never)]
pub fn foo_addr() -> usize {
    &FOO as *const bool as usize
}
