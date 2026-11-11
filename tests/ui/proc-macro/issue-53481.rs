//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

mod m1 {
    use crate::m2::Empty;

    #[derive(Empty)]
    struct A {}
}

mod m2 {
    pub type Empty = u8;

    #[derive(Empty)]
    #[empty_helper]
    struct B {}
}

fn main() {}
