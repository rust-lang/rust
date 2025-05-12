//@check-pass
//@revisions: old next
//@[next] compile-flags: -Znext-solver

#![feature(ptr_metadata)]
#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

use std::fmt::Debug;
use std::ptr::Thin;

fn check_thin<T: ?Sized + Thin>() {}

fn main() {
    check_thin::<dyn* Debug>();
}
