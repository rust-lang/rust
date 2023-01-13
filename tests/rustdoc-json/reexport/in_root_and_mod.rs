#![feature(no_core)]
#![no_core]

// @!has "$.index[*][?(@.name=='foo')]"
mod foo {
    // @has "$.index[*][?(@.name=='Foo')]"
    pub struct Foo;
}

// @has "$.index[*][?(@.kind=='import' && @.inner.source=='foo::Foo')]"
pub use foo::Foo;

pub mod bar {
    // @has "$.index[*][?(@.kind=='import' && @.inner.source=='crate::foo::Foo')]"
    pub use crate::foo::Foo;
}
