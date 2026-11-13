//@ check-pass

mod m {
    pub struct S(u8);

    use S as Z;
}

use m::*;

fn main() {}
