// check-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

#![feature(box_patterns)]

fn foo(box (_x, _y): Box<(isize, isize)>) {}

pub fn main() {}
