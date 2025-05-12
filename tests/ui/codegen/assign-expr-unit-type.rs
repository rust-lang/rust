//! Regression test for [Using the result of an assignment expression results in an LLVM assert
//! #483][issue-483]. This test checks that assignment expressions produce a unit type, and is
//! properly lowered to LLVM IR such that it does not trigger an LLVM assertion. This test was added
//! *really* early, back in 2011.
//!
//! [issue-483]: https://github.com/rust-lang/rust/issues/483

//@ run-pass

fn test_assign() {
    let mut x: isize;
    let y: () = x = 10;
    assert_eq!(x, 10);
    assert_eq!(y, ());
    let mut z: () = x = 11;
    assert_eq!(x, 11);
    assert_eq!(z, ());
    z = x = 12;
    assert_eq!(x, 12);
    assert_eq!(z, ());
}

fn test_assign_op() {
    let mut x: isize = 0;
    let y: () = x += 10;
    assert_eq!(x, 10);
    assert_eq!(y, ());
    let mut z: () = x += 11;
    assert_eq!(x, 21);
    assert_eq!(z, ());
    z = x += 12;
    assert_eq!(x, 33);
    assert_eq!(z, ());
}

pub fn main() {
    test_assign();
    test_assign_op();
}
