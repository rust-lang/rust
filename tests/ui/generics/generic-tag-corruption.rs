// run-pass
#![allow(non_camel_case_types)]


// This used to cause memory corruption in stage 0.
// pretty-expanded FIXME #23616

enum thing<K> { some(#[allow(unused_tuple_struct_fields)] K), }

pub fn main() { let _x = thing::some("hi".to_string()); }
