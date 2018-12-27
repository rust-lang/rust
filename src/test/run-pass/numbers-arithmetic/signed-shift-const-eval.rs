// run-pass
#![allow(non_camel_case_types)]


enum test { thing = -5 >> 1_usize }
pub fn main() {
    assert_eq!(test::thing as isize, -3);
}
