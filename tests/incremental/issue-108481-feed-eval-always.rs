//@ revisions: cpass1 cpass2

#![crate_type = "rlib"]

use std::fmt::Debug;

// MCVE kindly provided by Nilstrieb at
// https://github.com/rust-lang/rust/issues/108481#issuecomment-1493080185

#[derive(Debug)]
pub struct ConstGeneric<const CHUNK_SIZE: usize> {
    _p: [(); CHUNK_SIZE],
}

#[cfg(cpass1)]
impl<const CHUNK_SIZE: usize> ConstGeneric<CHUNK_SIZE> {}
