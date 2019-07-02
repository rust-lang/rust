// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
type A = for<> fn();

type B = for<'a,> fn();

pub fn main() {}
