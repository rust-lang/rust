#![feature(no_core)]
#![no_core]

pub mod foo {
    // @set bar_id = in_root_and_mod_pub.json "$.index[*][?(@.name=='Bar')].id"
    // @has - "$.index[*][?(@.name=='foo')].inner.items[*]" $bar_id
    pub struct Bar;
}

// @set root_import_id = - "$.index[*][?(@.inner.source=='foo::Bar')].id"
// @is - "$.index[*][?(@.inner.source=='foo::Bar')].inner.id" $bar_id
// @has - "$.index[*][?(@.name=='in_root_and_mod_pub')].inner.items[*]" $root_import_id
pub use foo::Bar;

pub mod baz {
    // @set baz_import_id = - "$.index[*][?(@.inner.source=='crate::foo::Bar')].id"
    // @is - "$.index[*][?(@.inner.source=='crate::foo::Bar')].inner.id" $bar_id
    // @has - "$.index[*][?(@.name=='baz')].inner.items[*]" $baz_import_id
    pub use crate::foo::Bar;
}
