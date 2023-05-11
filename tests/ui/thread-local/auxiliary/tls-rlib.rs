// no-prefer-dynamic

#![crate_type = "rlib"]
#![feature(thread_local)]
#![feature(cfg_target_thread_local)]

#[cfg(target_thread_local)]
#[thread_local]
pub static BAR: bool = true;

#[cfg(target_thread_local)]
#[inline(never)]
pub fn bar_addr() -> usize {
    &BAR as *const bool as usize
}
