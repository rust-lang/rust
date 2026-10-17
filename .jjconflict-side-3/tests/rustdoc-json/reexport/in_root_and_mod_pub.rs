pub mod foo {
    //@ set bar_id = "$.index[?(@.name=='Bar')].id"
    //@ ismany "$.index[?(@.name=='foo')].inner.module.items[*]" $bar_id
    pub struct Bar;
}

//@ set root_import_id = "$.index[?(@.docs=='Outer re-export')].id"
//@ is "$.index[?(@.inner.use.source=='foo::Bar')].inner.use.id" $bar_id
//@ has "$.index[?(@.name=='in_root_and_mod_pub')].inner.module.items[*]" $root_import_id
/// Outer re-export
pub use foo::Bar;

pub mod baz {
    //@ set baz_import_id = "$.index[?(@.docs=='Inner re-export')].id"
    //@ is "$.index[?(@.inner.use.source=='crate::foo::Bar')].inner.use.id" $bar_id
    //@ ismany "$.index[?(@.name=='baz')].inner.module.items[*]" $baz_import_id
    /// Inner re-export
    pub use crate::foo::Bar;
}
