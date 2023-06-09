#![feature(no_core)]
#![no_core]

// @!has "$.index[*][?(@.name=='foo')]"
mod foo {
    // @has "$.index[*][?(@.name=='Foo')]"
    pub struct Foo;
}

// @has "$.index[*].inner[?(@.import.source=='foo::Foo')]"
pub use foo::Foo;

pub mod bar {
    // @has "$.index[*].inner[?(@.import.source=='crate::foo::Foo')]"
    pub use crate::foo::Foo;
}
