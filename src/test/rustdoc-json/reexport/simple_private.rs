// edition:2018

#![no_core]
#![feature(no_core)]

// @!has simple_private.json "$.index[*][?(@.name=='inner')]"
// @set pub_id = - "$.index[*][?(@.name=='Public')].id"
// @has - "$.index[*][?(@.name=='simple_private')].inner.items[*]" $pub_id

mod inner {
    pub struct Public;
}
pub use inner::Public;
