//! Regression test for https://github.com/rust-lang/rust/issues/18809

//@ check-pass
trait Tup {
    type T0;
    type T1;
}

impl Tup for isize {
    type T0 = f32;
    type T1 = ();
}

fn main() {}
