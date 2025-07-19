//! Regression test for https://github.com/rust-lang/rust/issues/10767

//@ run-pass

pub fn main() {
    fn f() {
    }
    let _: Box<fn()> = Box::new(f as fn());
}
