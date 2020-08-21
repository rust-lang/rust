// aux-build:test-macros.rs
// check-pass
// compile-flags: -Z span-debug
// normalize-stdout-test "#\d+" -> "#CTXT"

extern crate test_macros;
use test_macros::print_bang;

macro_rules! use_expr {
    ($expr:expr) => {
        print_bang!($expr)
    }
}

macro_rules! use_pat {
    ($pat:pat) => {
        print_bang!($pat)
    }
}

#[allow(dead_code)]
struct Foo;
impl Foo {
    #[allow(dead_code)]
    fn use_self(self) {
        drop(use_expr!(self));
    }

    fn with_pat(use_pat!((a, b)): (u32, u32)) {
        println!("Args: {} {}", a, b);
    }
}

fn main() {}
