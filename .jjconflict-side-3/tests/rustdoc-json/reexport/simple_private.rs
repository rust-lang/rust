//@ edition:2018

//@ jq_count '.index[] | select(.name == "inner")' 0
mod inner {
    //@ jq_set pub_id = '.index[] | select(.name == "Public").id'
    pub struct Public;
}

//@ jq_is '.index[] | select(.inner.use).inner.use.name' '"Public"'
//@ jq_is '.index[] | select(.inner.use).inner.use.id' $pub_id
//@ jq_set use_id = '.index[] | select(.inner.use).id'
pub use inner::Public;

//@ jq_ismany '.index[] | select(.name == "simple_private").inner.module.items[]' $use_id

// Test for https://github.com/rust-lang/rust/issues/135309
//@ jq_has '.paths[] | select(.kind == "module").path' '["simple_private"]'
//@ !jq_has '.paths[].path' '["simple_private", "inner"]'
//@ jq_has '.paths[] | select(.kind == "struct").path' '["simple_private", "inner", "Public"]'
//@ !jq_has '.paths[].path' '["simple_private", "Public"]'
