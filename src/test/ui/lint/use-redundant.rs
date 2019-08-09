// build-pass (FIXME(62277): could be check-pass?)
#![warn(unused_imports)]

use crate::foo::Bar; //~ WARNING first import

mod foo {
    pub type Bar = i32;
}

fn baz() -> Bar {
    3
}

mod m1 { pub struct S {} }
mod m2 { pub struct S {} }

use m1::*;
use m2::*;

fn main() {
    use crate::foo::Bar; //~ WARNING redundant import
    let _a: Bar = 3;
    baz();

    use m1::S; //~ WARNING redundant import
    let _s = S {};
}
