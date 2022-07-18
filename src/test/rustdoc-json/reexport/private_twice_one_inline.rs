// aux-build:pub-struct.rs

// Test for the ICE in https://github.com/rust-lang/rust/issues/83057
// An external type re-exported with different attributes shouldn't cause an error

#![no_core]
#![feature(no_core)]

extern crate pub_struct as foo;

#[doc(inline)]
pub use foo::Foo;

pub mod bar {
    pub use foo::Foo;
}

// @count private_twice_one_inline.json "$.index[*][?(@.kind=='import')]" 2
