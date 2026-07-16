//! Regression test for <https://github.com/rust-lang/rust/issues/24954>.
//@ run-pass
macro_rules! foo {
    ($y:expr) => ({
        $y = 2;
    })
}

#[allow(unused_variables)]
#[allow(unused_assignments)]
fn main() {
    let mut x = 1;
    foo!(x);
}
