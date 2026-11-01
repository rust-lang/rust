//! System bindings for the Trusty OS.

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
mod common;

pub use common::*;
