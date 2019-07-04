// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

static x: &'static usize = &1;
static y: usize = *x;

fn main() {}
