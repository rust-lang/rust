//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use(Empty)]
extern crate test_macros;

use self::one::*;
use self::two::*;

mod empty_helper {}

mod one {
    use crate::empty_helper;

    #[derive(Empty)]
    #[empty_helper]
    struct One;
}

mod two {
    use crate::empty_helper;

    #[derive(Empty)]
    #[empty_helper]
    struct Two;
}

fn main() {}
