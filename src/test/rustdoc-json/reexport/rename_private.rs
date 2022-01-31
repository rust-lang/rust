// edition:2018

#![no_core]
#![feature(no_core)]

// @is rename_private.json "$.index[*][?(@.name=='inner')].kind" \"stripped_module\"
mod inner {
    // @has - "$.index[*][?(@.name=='Public')]"
    pub struct Public;
}

// @is - "$.index[*][?(@.kind=='import')].inner.name" \"NewName\"
pub use inner::Public as NewName;
