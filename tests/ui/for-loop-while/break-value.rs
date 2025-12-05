//@ run-pass
#![allow(unreachable_code)]

fn int_id(x: isize) -> isize { return x; }

pub fn main() { loop { int_id(break); } }
