// edition:2018
// compile-flags: --crate-version 1.0.0

// @is "$.crate_version" \"1.0.0\"
// @is "$.index[*][?(@.name=='nested')].kind" \"module\"
// @is "$.index[*][?(@.name=='nested')].inner.is_crate" true

// @set l1_id = "$.index[*][?(@.name=='l1')].id"
// @ismany "$.index[*][?(@.name=='nested')].inner.items[*]" $l1_id

// @is "$.index[*][?(@.name=='l1')].kind" \"module\"
// @is "$.index[*][?(@.name=='l1')].inner.is_crate" false
pub mod l1 {
    // @is "$.index[*][?(@.name=='l3')].kind" \"module\"
    // @is "$.index[*][?(@.name=='l3')].inner.is_crate" false
    // @set l3_id = "$.index[*][?(@.name=='l3')].id"
    pub mod l3 {

        // @is "$.index[*][?(@.name=='L4')].kind" \"struct\"
        // @is "$.index[*][?(@.name=='L4')].inner.kind" \"unit\"
        // @set l4_id = "$.index[*][?(@.name=='L4')].id"
        // @ismany "$.index[*][?(@.name=='l3')].inner.items[*]" $l4_id
        pub struct L4;
    }
    // @is "$.index[*][?(@.inner.source=='l3::L4')].kind" \"import\"
    // @is "$.index[*][?(@.inner.source=='l3::L4')].inner.glob" false
    // @is "$.index[*][?(@.inner.source=='l3::L4')].inner.id" $l4_id
    // @set l4_use_id = "$.index[*][?(@.inner.source=='l3::L4')].id"
    pub use l3::L4;
}
// @ismany "$.index[*][?(@.name=='l1')].inner.items[*]" $l3_id $l4_use_id
