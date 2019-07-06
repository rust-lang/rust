// build-pass (FIXME(62277): could be check-pass?)

mod m {
    pub struct S(u8);

    use S as Z;
}

use m::*;

fn main() {}
