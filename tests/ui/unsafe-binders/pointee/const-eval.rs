//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

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
