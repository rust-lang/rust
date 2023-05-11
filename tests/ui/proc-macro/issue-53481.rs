// build-pass (FIXME(62277): could be check-pass?)
// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

mod m1 {
    use m2::Empty;

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
