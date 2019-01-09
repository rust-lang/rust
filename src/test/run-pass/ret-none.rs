#![allow(non_camel_case_types)]
#![allow(dead_code)]


// pretty-expanded FIXME #23616

enum option<T> { none, some(T), }

fn f<T>() -> option<T> { return option::none; }

pub fn main() { f::<isize>(); }
