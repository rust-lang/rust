#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

const foo: isize = 4 >> 1;
enum bs { thing = foo }
pub fn main() { assert_eq!(bs::thing as isize, foo); }
