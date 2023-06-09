// edition:2018

#![no_core]
#![feature(no_core)]

// @!has "$.index[*][?(@.name=='inner')]"
mod inner {
    // @has "$.index[*][?(@.name=='Public')]"
    pub struct Public;
}

// @is "$.index[*][?(@.inner.import)].inner.import.name" \"NewName\"
pub use inner::Public as NewName;
