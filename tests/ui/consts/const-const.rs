//@ run-pass
#![allow(non_upper_case_globals)]

const a: isize = 1;
const b: isize = a + 2;

pub fn main() {
    assert_eq!(b, 3);
}
