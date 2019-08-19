// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

// pretty-expanded FIXME #23616

static mut n_mut: usize = 0;

static n: &'static usize = unsafe{ &n_mut };

fn main() {}
