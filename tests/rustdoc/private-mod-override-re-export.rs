// https://github.com/rust-lang/rust/issues/60926
#![crate_name = "foo"]
#![crate_type = "lib"]

mod m1 {
    pub mod m2 {
        pub struct Foo;
    }
}

pub use m1::*;
use crate::m1::m2;

//@ !has foo/index.html '//a[@href="m2/index.html"]' 'm2'
