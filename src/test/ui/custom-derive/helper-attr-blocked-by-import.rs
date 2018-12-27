// compile-pass
// aux-build:plugin.rs

#[macro_use(WithHelper)]
extern crate plugin;

use self::one::*;
use self::two::*;

mod helper {}

mod one {
    use helper;

    #[derive(WithHelper)]
    #[helper]
    struct One;
}

mod two {
    use helper;

    #[derive(WithHelper)]
    #[helper]
    struct Two;
}

fn main() {}
