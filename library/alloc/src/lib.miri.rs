//! Grep bootstrap for `MIRI_REPLACE_LIBRS_IF_NOT_TEST` to learn what this is about.
#![no_std]
extern crate alloc as realalloc;
pub use realalloc::*;
