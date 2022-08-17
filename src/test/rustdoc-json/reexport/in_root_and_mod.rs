#![feature(no_core)]
#![no_core]

// @is "$.index[*][?(@.name=='foo')].kind" \"module\"
// @is "$.index[*][?(@.name=='foo')].inner.is_stripped" "true"
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
