//@ edition:2018

//@ !has "$.index[?(@.name=='inner')]"
mod inner {
    //@ set pub_id = "$.index[?(@.name=='Public')].id"
    pub struct Public;
}

//@ is "$.index[?(@.inner.use)].inner.use.name" \"Public\"
//@ is "$.index[?(@.inner.use)].inner.use.id" $pub_id
//@ set use_id = "$.index[?(@.inner.use)].id"
pub use inner::Public;

//@ ismany "$.index[?(@.name=='simple_private')].inner.module.items[*]" $use_id

// Test for https://github.com/rust-lang/rust/issues/135309
//@ has  "$.paths[?(@.kind=='module')].path" '["simple_private"]'
//@ !has "$.paths[*].path"                      '["simple_private", "inner"]'
//@ has  "$.paths[?(@.kind=='struct')].path" '["simple_private", "inner", "Public"]'
//@ !has "$.paths[*].path"                      '["simple_private", "Public"]'
