//@ check-pass
#![allow(dead_code)]

#![feature(box_patterns)]

fn foo(box (_x, _y): Box<(isize, isize)>) {}

pub fn main() {}
