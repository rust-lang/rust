// edition:2018
#![no_core]
#![feature(no_core)]

// @is "$.index[*][?(@.name=='inner')].kind" \"module\"
// @is "$.index[*][?(@.name=='inner')].inner.is_stripped" "true"
mod inner {
    // @set pub_id = "$.index[*][?(@.name=='Public')].id"
    pub struct Public;
}

// @is "$.index[*][?(@.kind=='import')].inner.name" \"Public\"
// @set use_id = "$.index[*][?(@.kind=='import')].id"
pub use inner::Public;

// @ismany "$.index[*][?(@.name=='inner')].inner.items[*]" $pub_id
// @ismany "$.index[*][?(@.name=='simple_private')].inner.items[*]" $use_id
