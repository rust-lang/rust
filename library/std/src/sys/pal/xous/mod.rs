#![forbid(unsafe_op_in_unsafe_fn)]

use crate::os::xous::ffi::exit;

pub mod os;
pub mod time;

#[path = "../unsupported/common.rs"]
mod common;
pub use common::*;

pub fn abort_internal() -> ! {
    exit(101);
}
