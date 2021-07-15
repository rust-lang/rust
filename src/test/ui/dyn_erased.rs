// build-fail
// compile-flags: -Zmir-opt-level=1

#![feature(rustc_attrs)]
#![feature(core_intrinsics)]

use std::fmt::Display;

#[rustc_dyn]
fn intrinsic<T>(f: *mut T) -> *const T {
    //~^ ERROR rustc_dyn: unhandled intrinsic
    let f = unsafe { std::intrinsics::offset(f, 5) };
    f as *const _
}

#[rustc_dyn]
fn bad_niche<T>(f: &Option<T>) -> &T {
    //~^ ERROR rustc_dyn: unknown layout
    if let Some(f) = f { f } else { panic!() }
}

#[rustc_dyn]
fn fat_ptr<T: ?Sized>(f: &T) -> &T {
    //~^ ERROR rustc_dyn: unknown layout
    f
}

fn main() {
    intrinsic(&mut 42 as _);
    bad_niche(&Some(42));
    fat_ptr(&42);
}
