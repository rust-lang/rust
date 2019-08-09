// build-pass (FIXME(62277): could be check-pass?)
// aux-build:test-macros.rs

extern crate test_macros;

mod inner {
    use test_macros::Empty;

    #[derive(Empty)]
    struct S;
}

fn main() {}
