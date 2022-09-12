// edition:2018

#![no_core]
#![feature(no_core)]

// @set inner_id = "$.index[*][?(@.name=='inner')].id"
pub mod inner {
    // @set public_id = "$.index[*][?(@.name=='Public')].id"
    // @ismany "$.index[*][?(@.name=='inner')].inner.items[*]" $public_id
    pub struct Public;
}
// @set import_id = "$.index[*][?(@.inner.name=='NewName')].id"
// @!has "$.index[*][?(@.inner.name=='Public')]"
// @is "$.index[*][?(@.inner.name=='NewName')].inner.source" \"inner::Public\"
pub use inner::Public as NewName;

// @ismany "$.index[*][?(@.name=='rename_public')].inner.items[*]" $inner_id $import_id
