//@ edition:2018

//@ !has "$.index[*][?(@.name=='inner')]"
mod inner {
    //@ set pub_id = "$.index[*][?(@.name=='Public')].id"
    pub struct Public;
}

//@ is "$.index[*][?(@.inner.use)].inner.use.name" \"Public\"
//@ is "$.index[*][?(@.inner.use)].inner.use.id" $pub_id
//@ set use_id = "$.index[*][?(@.inner.use)].id"
pub use inner::Public;

//@ ismany "$.index[*][?(@.name=='simple_private')].inner.module.items[*]" $use_id
