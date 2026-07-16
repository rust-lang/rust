#![allow(unused)]
#![deny(c_void_returns)]

use std::ffi::c_void;
use std::ptr;

fn foo() -> c_void { //~ ERROR c_void
    unreachable!()
}

fn bar() -> *mut c_void {
    ptr::null_mut()
}

unsafe extern "C" {
    fn baz() -> c_void; //~ ERROR c_void
    fn quux() -> *const c_void;
}

type Xyzzy = fn() -> c_void; //~ ERROR c_void

fn main() {}
