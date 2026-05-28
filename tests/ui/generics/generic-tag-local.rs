//@ run-pass
#![allow(non_camel_case_types)]


enum clam<T> { a(#[allow(dead_code)] T), }

pub fn main() { let _c = clam::a(3); }
