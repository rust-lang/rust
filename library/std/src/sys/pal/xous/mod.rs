#![forbid(unsafe_op_in_unsafe_fn)]

pub mod os;

#[path = "../unsupported/common.rs"]
mod common;
pub use common::*;
