#![allow(non_camel_case_types)]
#![allow(dead_code)]


// pretty-expanded FIXME #23616

enum option<T> { some(T), none, }

struct R<T> {v: Vec<option<T>> }

fn f<T>() -> Vec<T> { return Vec::new(); }

pub fn main() { let mut r: R<isize> = R {v: Vec::new()}; r.v = f(); }
