//@ run-pass
#![allow(non_camel_case_types)]


// This used to cause memory corruption in stage 0.

enum thing<K> { some(#[allow(dead_code)] K), }

pub fn main() { let _x = thing::some("hi".to_string()); }
