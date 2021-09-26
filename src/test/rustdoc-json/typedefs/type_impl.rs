#![no_std]

// @has type_impl.json "$.index[*][?(@.name=='Ix')].visibility" \"public\"
// @has - "$.index[*][?(@.name=='Ix')].kind" \"typedef\"
pub type Ix = usize;

// @has - "$.index[*][?(@.name=='IxTrait')].visibility" \"public\"
// @has - "$.index[*][?(@.name=='IxTrait')].kind" \"trait\"
pub trait IxTrait {}

// @count - "$.index[*][?(@.name=='Ix')].inner.impls" 1
impl IxTrait for Ix {}
