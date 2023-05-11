// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

use std::ptr;

fn main() {
    (&ptr::write)(1 as *mut _, 42);
    //~^ ERROR E0133
}
