#![crate_type = "dylib"]
#![feature(thread_local)]
#![feature(cfg_target_thread_local)]
#![cfg(target_thread_local)]

extern crate tls_rlib;

pub use tls_rlib::*;

#[thread_local]
pub static FOO: bool = true;

#[inline(never)]
pub fn foo_addr() -> usize {
    &FOO as *const bool as usize
}
