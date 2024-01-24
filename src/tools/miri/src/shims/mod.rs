#![warn(clippy::arithmetic_side_effects)]

mod backtrace;
#[cfg(target_os = "linux")]
pub mod ffi_support;
pub mod foreign_items;
pub mod intrinsics;
pub mod unix;
pub mod windows;
mod x86;

pub mod env;
pub mod extern_static;
pub mod os_str;
pub mod panic;
pub mod time;
pub mod tls;
