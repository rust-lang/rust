//@ check-fail
//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(rustc_attrs)]

#[rustc_force_inline]
pub fn callee(x: isize) -> usize { unimplemented!() }

fn a() {
    let _: fn(isize) -> usize = callee;
//~^ ERROR cannot coerce functions which must be inlined to function pointers
}

fn b() {
    let _ = callee as fn(isize) -> usize;
//~^ ERROR non-primitive cast
}

fn c() {
    let _: [fn(isize) -> usize; 2] = [
        callee,
//~^ ERROR cannot coerce functions which must be inlined to function pointers
        callee,
    ];
}
