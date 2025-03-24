//@ run-pass
// Issue #2040

#![allow(unnecessary_refs)]

pub fn main() {
    let foo: isize = 1;
    assert_eq!(&foo as *const isize, &foo as *const isize);
}
