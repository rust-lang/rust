// Original implementation taken from rust-memchr.
// Copyright 2015 Andrew Gallant, bluss and Nicolas Koch

// Fallback memchr is fastest on Windows.
#![deny(unsafe_op_in_unsafe_fn)]
pub use core::slice::memchr::{memchr, memrchr};
