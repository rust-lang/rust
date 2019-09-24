// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

// pretty-expanded FIXME #23616

static foo: [usize; 3] = [1, 2, 3];

static slice_1: &'static [usize] = &foo;
static slice_2: &'static [usize] = &foo;

fn main() {}
