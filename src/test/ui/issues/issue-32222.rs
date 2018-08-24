#![feature(rustc_attrs)]
#![allow(warnings)]

mod foo {
    pub fn bar() {}
}

pub use foo::*;
use b::bar;

mod foobar {
    use super::*;
}

mod a {
    pub mod bar {}
}

mod b {
    pub use a::bar;
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
