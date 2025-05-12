//@ run-pass
//@ aux-build:some_crate.rs
//@ compile-flags:--extern some_crate
//@ edition:2018

use some_crate as some_name;

mod foo {
    pub use crate::some_name::*;
}

fn main() {
    ::some_crate::hello();
    some_name::hello();
    foo::hello();
}
