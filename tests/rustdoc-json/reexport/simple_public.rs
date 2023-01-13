// edition:2018

#![no_core]
#![feature(no_core)]

// @set inner_id = "$.index[*][?(@.name=='inner')].id"
pub mod inner {

    // @set public_id = "$.index[*][?(@.name=='Public')].id"
    // @ismany "$.index[*][?(@.name=='inner')].inner.items[*]" $public_id
    pub struct Public;
}

// @set import_id = "$.index[*][?(@.inner.name=='Public')].id"
// @is "$.index[*][?(@.inner.name=='Public')].inner.source" \"inner::Public\"
pub use inner::Public;

// @ismany "$.index[*][?(@.name=='simple_public')].inner.items[*]" $import_id $inner_id
