#![forbid(unsafe_op_in_unsafe_fn)]

pub mod os;
pub mod time;

#[path = "../unsupported/common.rs"]
mod common;
pub use common::*;
