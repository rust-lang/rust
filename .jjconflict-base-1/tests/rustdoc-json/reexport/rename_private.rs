//@ edition:2018

//@ !has "$.index[?(@.name=='inner')]"
mod inner {
    //@ has "$.index[?(@.name=='Public')]"
    pub struct Public;
}

//@ is "$.index[?(@.inner.use)].inner.use.name" \"NewName\"
pub use inner::Public as NewName;
