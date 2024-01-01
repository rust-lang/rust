// build-pass

#![allow(non_camel_case_types)]



// pretty-expanded FIXME #23616

type lteq<T> = extern "C" fn(T) -> bool;

pub fn main() { }
