//
// Original implementation taken from rust-memchr
// Copyright 2015 Andrew Gallant, bluss and Nicolas Koch

// Fallback memchr is fastest on windows
pub use core::slice::memchr::{memchr, memrchr};
