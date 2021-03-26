// edition:2018

#![no_core]
#![feature(no_core)]

// @set inner_id = simple_public.json "$.index[*][?(@.name=='inner')].id"
// @has - "$.index[*][?(@.name=='simple_public')].inner.items[*]" $inner_id
pub mod inner {

    // @set public_id = - "$.index[*][?(@.name=='Public')].id"
    // @has - "$.index[*][?(@.name=='inner')].inner.items[*]" $public_id
    pub struct Public;
}

// @set import_id = - "$.index[*][?(@.inner.name=='Public')].id"
// @has - "$.index[*][?(@.name=='simple_public')].inner.items[*]" $import_id
// @is - "$.index[*][?(@.inner.name=='Public')].inner.source" \"inner::Public\"
pub use inner::Public;
