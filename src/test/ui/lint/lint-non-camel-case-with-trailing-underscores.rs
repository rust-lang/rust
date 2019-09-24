// build-pass (FIXME(62277): could be check-pass?)

#![allow(dead_code)]
// This is ok because we often use the trailing underscore to mean 'prime'

// pretty-expanded FIXME #23616

#[forbid(non_camel_case_types)]
type Foo_ = isize;

pub fn main() { }
