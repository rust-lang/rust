//@ build-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]


static foo: [usize; 3] = [1, 2, 3];

static slice_1: &'static [usize] = &foo;
static slice_2: &'static [usize] = &foo;

fn main() {}
