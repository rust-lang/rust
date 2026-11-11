//@ build-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

static mut n_mut: usize = 0;

static n: &'static usize = unsafe { &n_mut };
//~^ WARN shared reference to mutable static [static_mut_refs]

fn main() {}
