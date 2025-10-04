#![allow(unsafe_op_in_unsafe_fn)]

pub mod os;
pub mod pipe;
pub mod time;

pub use moto_rt::futex;

#[path = "../unsupported/common.rs"]
mod common;
pub use common::*;

#[cfg(not(test))]
#[unsafe(no_mangle)]
pub extern "C" fn motor_start() -> ! {
    // Initialize the runtime.
    moto_rt::start();

    // Call main.
    unsafe extern "C" {
        fn main(_: isize, _: *const *const u8, _: u8) -> i32;
    }
    let result = unsafe { main(0, core::ptr::null(), 0) };

    // Terminate the process.
    moto_rt::process::exit(result)
}

pub(crate) use crate::os::motor::map_motor_error;
