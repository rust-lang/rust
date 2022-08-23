// edition:2018

#![no_core]
#![feature(no_core)]

// @is "$.index[*][?(@.name=='inner')].kind" \"module\"
// @is "$.index[*][?(@.name=='inner')].inner.is_stripped" "true"
mod inner {
    // @has "$.index[*][?(@.name=='Public')]"
    pub struct Public;
}

// @is "$.index[*][?(@.kind=='import')].inner.name" \"NewName\"
pub use inner::Public as NewName;
