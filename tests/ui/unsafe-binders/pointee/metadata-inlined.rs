//@ compile-flags: -Znext-solver -O
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// Pointer metadata of a binder whose tail is an alias mentioning the bound
// lifetime works after the MIR inliner runs (`-O`).

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features)]

use std::mem::ManuallyDrop as MD;
use std::ptr::{self, Pointee};

trait Assoc {
    type Assoc<'a>: ?Sized;
}

struct Slice;
impl Assoc for Slice {
    type Assoc<'a> = [&'a u8];
}

#[inline]
fn meta<T: Assoc>(
    p: *const unsafe<'a> MD<T::Assoc<'a>>,
) -> <unsafe<'a> MD<T::Assoc<'a>> as Pointee>::Metadata {
    ptr::metadata(p)
}

fn main() {
    let arr: [&u8; 2] = [&1, &2];
    let s: &[&u8] = &arr;
    let p = s as *const [&u8] as *const unsafe<'a> MD<[&'a u8]>;
    assert_eq!(meta::<Slice>(p), 2);
}
