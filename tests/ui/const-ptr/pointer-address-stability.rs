//! Check that taking the address of a stack variable with `&`
//! yields a stable and comparable pointer.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/2040>.

//@ run-pass

pub fn main() {
    let foo: isize = 1;
    assert_eq!(&foo as *const isize, &foo as *const isize);
}
