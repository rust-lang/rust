#![forbid(unsafe_op_in_unsafe_fn)]

pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
pub mod thread;
pub mod time;

#[path = "../unsupported/common.rs"]
mod common;
pub use common::*;
