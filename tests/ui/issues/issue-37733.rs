//@ build-pass
#![allow(dead_code)]
type A = for<> fn();

type B = for<'a,> fn();

pub fn main() {}
