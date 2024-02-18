//@ run-pass

#[allow(unused_macro_rules)]
macro_rules! a {
    ($i:literal) => { "right" };
    ($i:tt) => { "wrong" };
}

macro_rules! b {
    ($i:literal) => { a!($i) };
}

fn main() {
    assert_eq!(b!(0), "right");
}
