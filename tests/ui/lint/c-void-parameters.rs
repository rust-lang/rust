//@ revisions: direct alias

#![allow(unused)]
#![deny(c_void_parameters)]

#[cfg(direct)]
use std::ffi::c_void;
#[cfg(alias)]
#[expect(non_camel_case_types)]
type c_void = std::ffi::c_void;

use std::ptr;

fn foo(_: c_void) {
    //~^ ERROR c_void
}

fn bar(_: *mut c_void) {}

unsafe extern "C" {
    fn baz(_: c_void); //~ ERROR c_void
    fn quux(_: *const c_void);
}

type Xyzzy = fn(c_void); //~ ERROR c_void

trait Trait {
    fn foo(_: c_void); //~ ERROR c_void
}

fn main() {}
