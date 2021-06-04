// edition:2018

#![no_core]
#![feature(no_core)]

// @!has simple_private.json "$.index[*][?(@.name=='inner')]"
mod inner {
    // @set pub_id = - "$.index[*][?(@.name=='Public')].id"
    pub struct Public;
}

// @has - "$.index[*][?(@.name=='simple_private')].inner.items[*]" $pub_id
pub use inner::Public;
