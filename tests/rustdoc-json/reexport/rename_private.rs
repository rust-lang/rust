// edition:2018

#![no_core]
#![feature(no_core)]

// @!has "$.index[*][?(@.kind=='inner')]"
mod inner {
    // @has "$.index[*][?(@.name=='Public')]"
    pub struct Public;
}

// @is "$.index[*][?(@.kind=='import')].inner.name" \"NewName\"
pub use inner::Public as NewName;
