//! Regression test for https://github.com/rust-lang/rust/issues/15189

//@ run-pass
macro_rules! third {
    ($e:expr) => ({let x = 2; $e[x]})
}

fn main() {
    let x = vec![10_usize,11_usize,12_usize,13_usize];
    let t = third!(x);
    assert_eq!(t,12_usize);
}
