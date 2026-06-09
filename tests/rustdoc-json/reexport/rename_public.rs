//@ edition:2018

//@ set inner_id = "$.index[?(@.name=='inner')].id"
pub mod inner {
    //@ set public_id = "$.index[?(@.name=='Public')].id"
    //@ ismany "$.index[?(@.name=='inner')].inner.module.items[*]" $public_id
    pub struct Public;
}
//@ set import_id = "$.index[?(@.docs=='Re-export')].id"
//@ !has "$.index[?(@.inner.use.name=='Public')]"
//@ is "$.index[?(@.inner.use.name=='NewName')].inner.use.source" \"inner::Public\"
/// Re-export
pub use inner::Public as NewName;

//@ ismany "$.index[?(@.name=='rename_public')].inner.module.items[*]" $inner_id $import_id
