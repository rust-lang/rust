//@ run-pass
#![allow(non_camel_case_types)]

//@ pretty-expanded FIXME #23616

enum clam<T> { a(#[allow(dead_code)] T), }

pub fn main() { let _c = clam::a(3); }
