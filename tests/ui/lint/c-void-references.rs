//@ revisions: direct alias

#![allow(unused)]
#![deny(c_void_references)]

#[cfg(direct)]
use std::ffi::c_void;
#[cfg(alias)]
#[expect(non_camel_case_types)]
type c_void = std::ffi::c_void;

fn foo() -> *mut c_void {
    // fine
    std::ptr::null_mut()
}

fn bar() -> &'static mut c_void {
    //~^ ERROR c_void
    panic!();
}

fn baz() -> Option<&'static c_void> {
    //~^ ERROR c_void
    None
}

fn quux(_: &c_void) {} //~ ERROR c_void

type Boo<'a> = &'a c_void;
//~^ ERROR c_void
fn main() {
    let _: &'static c_void = panic!();
    //~^ ERROR c_void
}

trait Trait {}

impl<'a> Trait for &'a c_void {}
//~^ ERROR c_void
