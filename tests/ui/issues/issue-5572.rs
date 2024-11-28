//@ check-pass
#![allow(dead_code)]

fn foo<T: ::std::cmp::PartialEq>(_t: T) { }

pub fn main() { }
