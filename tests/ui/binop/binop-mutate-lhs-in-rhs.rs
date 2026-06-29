//! Regression test for <https://github.com/rust-lang/rust/issues/27054>.
//@ run-pass

fn main() {
    let x = &mut 1;
    assert_eq!(*x + { *x=2; 1 }, 2);
}
