//! Grep bootstrap for `MIRI_REPLACE_LIBRS_IF_NOT_TEST` to learn what this is about.
#![no_std]
#![feature(rustc_private)]
extern crate compiler_builtins as real;
pub use real::*;
