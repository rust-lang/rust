// https://github.com/rust-lang/rust/issues/60926
#![crate_name = "foo"]

mod m1 {
    pub mod m2 {
        pub struct Foo;
    }
}

pub use m1::*;
use crate::m1::m2;

//@ count foo/index.html '//a[@class="mod"]' 0
