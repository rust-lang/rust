//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use(Empty)]
extern crate test_macros;

use self::one::*;
use self::two::*;

mod empty_helper {}

mod one {
    use empty_helper;

    #[derive(Empty)]
    #[empty_helper]
    struct One;
}

mod two {
    use empty_helper;

    #[derive(Empty)]
    #[empty_helper]
    struct Two;
}

fn main() {}
