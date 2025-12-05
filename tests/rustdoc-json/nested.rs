//@ edition:2018
//@ compile-flags: --crate-version 1.0.0

//@ is "$.crate_version" \"1.0.0\"
//@ has "$.index[?(@.name=='nested')].inner.module"
//@ is "$.index[?(@.name=='nested')].inner.module.is_crate" true

//@ set l1_id = "$.index[?(@.name=='l1')].id"
//@ ismany "$.index[?(@.name=='nested')].inner.module.items[*]" $l1_id

//@ has "$.index[?(@.name=='l1')].inner.module"
//@ is "$.index[?(@.name=='l1')].inner.module.is_crate" false
pub mod l1 {
    //@ has "$.index[?(@.name=='l3')].inner.module"
    //@ is "$.index[?(@.name=='l3')].inner.module.is_crate" false
    //@ set l3_id = "$.index[?(@.name=='l3')].id"
    pub mod l3 {

        //@ has "$.index[?(@.name=='L4')].inner.struct"
        //@ is "$.index[?(@.name=='L4')].inner.struct.kind" '"unit"'
        //@ set l4_id = "$.index[?(@.name=='L4')].id"
        //@ ismany "$.index[?(@.name=='l3')].inner.module.items[*]" $l4_id
        pub struct L4;
    }
    //@ is "$.index[?(@.inner.use)].inner.use.is_glob" false
    //@ is "$.index[?(@.inner.use)].inner.use.source" '"l3::L4"'
    //@ is "$.index[?(@.inner.use)].inner.use.is_glob" false
    //@ is "$.index[?(@.inner.use)].inner.use.id" $l4_id
    //@ set l4_use_id = "$.index[?(@.inner.use)].id"
    pub use l3::L4;
}
//@ ismany "$.index[?(@.name=='l1')].inner.module.items[*]" $l3_id $l4_use_id
