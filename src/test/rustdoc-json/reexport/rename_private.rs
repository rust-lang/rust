// edition:2018

#![no_core]
#![feature(no_core)]
// @!has rename_private.json "$.index[*][?(@.name=='inner')]"
mod inner {
    // @!has - "$.index[*][?(@.name=='Public')]"
    pub struct Public;
}

// @set newname_id = - "$.index[*][?(@.name=='NewName')].id"
// @is - "$.index[*][?(@.name=='NewName')].kind" \"struct\"
// @has - "$.index[*][?(@.name=='rename_private')].inner.items[*]" $newname_id
pub use inner::Public as NewName;
