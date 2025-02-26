//@ run-pass

#![allow(non_camel_case_types)]
#![allow(dead_code)]



type lteq<T> = extern "C" fn(T) -> bool;

pub fn main() { }
