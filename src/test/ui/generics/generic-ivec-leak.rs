// run-pass
#![allow(non_camel_case_types)]
enum wrapper<T> { wrapped(#[allow(unused_tuple_struct_fields)] T), }

pub fn main() { let _w = wrapper::wrapped(vec![1, 2, 3, 4, 5]); }
