//@ check-pass
#![warn(unused_imports, redundant_imports)]

use crate::foo::Bar;

mod foo {
    pub type Bar = i32;
}

fn baz() -> Bar {
    3
}

mod m1 { pub struct S {} }
mod m2 { pub struct S {} }

use m1::*; //~ WARNING unused import
use m2::*; //~ WARNING unused import

fn main() {
    use crate::foo::Bar; //~ WARNING imported redundantly
    let _a: Bar = 3;
    baz();

    use m1::S;
    let _s = S {};
}
