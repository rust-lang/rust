// edition:2018

#![no_core]
#![feature(no_core)]

// @set inner_id = "$.index[*][?(@.name=='inner')].id"
pub mod inner {

    // @set public_id = "$.index[*][?(@.name=='Public')].id"
    // @ismany "$.index[*][?(@.name=='inner')].inner.module.items[*]" $public_id
    pub struct Public;
}

// @set import_id = "$.index[*][?(@.docs=='Outer')].id"
// @is "$.index[*][?(@.docs=='Outer')].inner.import.source" \"inner::Public\"
/// Outer
pub use inner::Public;

// @ismany "$.index[*][?(@.name=='simple_public')].inner.module.items[*]" $import_id $inner_id
