// edition:2018
#![no_core]
#![feature(no_core)]

// @is simple_private.json "$.index[*][?(@.name=='inner')].kind" \"module\"
// @is simple_private.json "$.index[*][?(@.name=='inner')].inner.is_stripped" "true"
mod inner {
    // @set pub_id = - "$.index[*][?(@.name=='Public')].id"
    pub struct Public;
}

// @is - "$.index[*][?(@.kind=='import')].inner.name" \"Public\"
pub use inner::Public;

// @has - "$.index[*][?(@.name=='inner')].inner.items[*]" $pub_id
