// edition:2018
// compile-flags: --crate-version 1.0.0

// @is nested.json "$.crate_version" \"1.0.0\"
// @is - "$.index[*][?(@.name=='nested')].kind" \"module\"
// @is - "$.index[*][?(@.name=='nested')].inner.is_crate" true
// @count - "$.index[*][?(@.name=='nested')].inner.items[*]" 1

// @is nested.json "$.index[*][?(@.name=='l1')].kind" \"module\"
// @is - "$.index[*][?(@.name=='l1')].inner.is_crate" false
// @count - "$.index[*][?(@.name=='l1')].inner.items[*]" 2
pub mod l1 {

    // @is nested.json "$.index[*][?(@.name=='l3')].kind" \"module\"
    // @is - "$.index[*][?(@.name=='l3')].inner.is_crate" false
    // @count - "$.index[*][?(@.name=='l3')].inner.items[*]" 1
    // @set l3_id = - "$.index[*][?(@.name=='l3')].id"
    // @has - "$.index[*][?(@.name=='l1')].inner.items[*]" $l3_id
    pub mod l3 {

        // @is nested.json "$.index[*][?(@.name=='L4')].kind" \"struct\"
        // @is - "$.index[*][?(@.name=='L4')].inner.struct_type" \"unit\"
        // @set l4_id = - "$.index[*][?(@.name=='L4')].id"
        // @has - "$.index[*][?(@.name=='l3')].inner.items[*]" $l4_id
        pub struct L4;
    }
    // @is nested.json "$.index[*][?(@.inner.span=='l3::L4')].kind" \"import\"
    // @is - "$.index[*][?(@.inner.span=='l3::L4')].inner.glob" false
    pub use l3::L4;
}
