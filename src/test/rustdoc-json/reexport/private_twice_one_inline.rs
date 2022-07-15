// aux-build:pub-struct.rs

// Test for the ICE in rust/83057
// Am external type re-exported with different attributes shouldn't cause an error

#![no_core]
#![feature(no_core)]

extern crate pub_struct as foo;

#[doc(inline)]
pub use foo::Foo;

pub mod bar {
    pub use foo::Foo;
}

// @count private_twice_one_inline.json "$.index[*][?(@.kind=='import')]" 2
