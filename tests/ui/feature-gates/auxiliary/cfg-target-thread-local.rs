#![feature(thread_local)]
#![feature(cfg_target_thread_local)]
#![crate_type = "lib"]

#[no_mangle]
#[cfg_attr(target_thread_local, thread_local)]
pub static FOO: u32 = 3;
