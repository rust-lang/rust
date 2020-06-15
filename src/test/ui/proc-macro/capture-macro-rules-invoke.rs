// aux-build:test-macros.rs
// check-pass

extern crate test_macros;
use test_macros::recollect;

macro_rules! use_expr {
    ($expr:expr) => {
        recollect!($expr)
    }
}

#[allow(dead_code)]
struct Foo;
impl Foo {
    #[allow(dead_code)]
    fn use_self(self) {
        drop(use_expr!(self));
    }
}

fn main() {}
