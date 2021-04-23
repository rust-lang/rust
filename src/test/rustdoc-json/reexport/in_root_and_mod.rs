#![feature(no_core)]
#![no_core]

mod foo {
    // @set foo_id = in_root_and_mod.json "$.index[*][?(@.name=='Foo')].id"
    pub struct Foo;
}

// @has - "$.index[*][?(@.name=='in_root_and_mod')].inner.items[*]" $foo_id
pub use foo::Foo;

pub mod bar {
    // @has - "$.index[*][?(@.name=='bar')].inner.items[*]" $foo_id
    pub use crate::foo::Foo;
}
