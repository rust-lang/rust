//@ check-pass

#![allow(dead_code)]
// This is ok because we often use the trailing underscore to mean 'prime'


#[forbid(non_camel_case_types)]
type Foo_ = isize;

pub fn main() { }
