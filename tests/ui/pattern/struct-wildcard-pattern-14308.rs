//! Regression test for https://github.com/rust-lang/rust/issues/14308

//@ run-pass

struct A(isize);

fn main() {
    let x = match A(3) {
        A(..) => 1
    };
    assert_eq!(x, 1);
    let x = match A(4) {
        A(1) => 1,
        A(..) => 2
    };
    assert_eq!(x, 2);
}
