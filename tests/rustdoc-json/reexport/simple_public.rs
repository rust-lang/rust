//@ edition:2018

//@ set inner_id = "$.index[?(@.name=='inner')].id"
pub mod inner {

    //@ set public_id = "$.index[?(@.name=='Public')].id"
    //@ ismany "$.index[?(@.name=='inner')].inner.module.items[*]" $public_id
    pub struct Public;
}

//@ set import_id = "$.index[?(@.docs[0].text=='Outer')].id"
//@ is "$.index[?(@.docs[0].text=='Outer')].inner.use.source" \"inner::Public\"
/// Outer
pub use inner::Public;

//@ ismany "$.index[?(@.name=='simple_public')].inner.module.items[*]" $import_id $inner_id

//@ has  "$.paths[?(@.kind=='module')].path" '["simple_public"]'
//@ has  "$.paths[?(@.kind=='module')].path" '["simple_public", "inner"]'
//@ has  "$.paths[?(@.kind=='struct')].path" '["simple_public", "inner", "Public"]'
//@ !has "$.paths[*].path"                      '["simple_public", "Public"]'
