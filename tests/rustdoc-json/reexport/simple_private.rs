// edition:2018
#![no_core]
#![feature(no_core)]

// @!has "$.index[*][?(@.name=='inner')]"
mod inner {
    // @set pub_id = "$.index[*][?(@.name=='Public')].id"
    pub struct Public;
}

// @is "$.index[*][?(@.inner.import)].inner.import.name" \"Public\"
// @is "$.index[*][?(@.inner.import)].inner.import.id" $pub_id
// @set use_id = "$.index[*][?(@.inner.import)].id"
pub use inner::Public;

// @ismany "$.index[*][?(@.name=='simple_private')].inner.module.items[*]" $use_id
