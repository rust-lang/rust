//@ revisions: rpass cfail
//@ ignore-backends: gcc

#![deny(unused_features)]

// Used language features
#![feature(box_patterns)]
#![feature(decl_macro)]
#![cfg_attr(all(), feature(rustc_attrs))]

// Used library features
#![feature(error_iter)]
//[cfail]~^ ERROR feature `error_iter` is declared but not used
#![cfg_attr(all(), feature(allocator_api))]
//[cfail]~^ ERROR feature `allocator_api` is declared but not used

pub fn use_box_patterns(b: Box<i32>) -> i32 {
    let box x = b;
    x
}

macro m() {}
pub fn use_decl_macro() {
    m!();
}

#[rustc_dummy]
pub fn use_rustc_attrs() {}

#[cfg(rpass)]
pub fn use_error_iter(e: &(dyn std::error::Error + 'static)) {
    for _ in e.sources() {}
}

#[cfg(rpass)]
pub fn use_allocator_api() {
    use std::alloc::Global;
    let _ = Vec::<i32>::new_in(Global);
}

fn main() {}
