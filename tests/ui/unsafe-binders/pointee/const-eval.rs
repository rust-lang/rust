//@ compile-flags: -Znext-solver
//@ check-pass

// Const-eval computes the metadata and dynamic size of binders on its own path.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::mem::{ManuallyDrop as MD, size_of_val};
use std::ptr;

const S: &[u8] = &[1, 2, 3];
const P: *const unsafe<> MD<[u8]> = S as *const [u8] as *const MD<[u8]> as *const _;

const LEN: usize = ptr::metadata(P);
const _: () = assert!(LEN == 3);
const SIZE: usize = unsafe { size_of_val(&*P) };
const _: () = assert!(SIZE == 3);

fn main() {}
