// edition:2018

// @has nested.json "$.index[*][?(@.name=='nested')].kind" \"module\"
// @has - "$.index[*][?(@.name=='nested')].inner.is_crate" true
// @count - "$.index[*][?(@.name=='nested')].inner.items[*]" 1

// @has nested.json "$.index[*][?(@.name=='l1')].kind" \"module\"
// @has - "$.index[*][?(@.name=='l1')].inner.is_crate" false
// @count - "$.index[*][?(@.name=='l1')].inner.items[*]" 2
pub mod l1 {

    // @has nested.json "$.index[*][?(@.name=='l3')].kind" \"module\"
    // @has - "$.index[*][?(@.name=='l3')].inner.is_crate" false
    // @count - "$.index[*][?(@.name=='l3')].inner.items[*]" 1
    pub mod l3 {

        // @has nested.json "$.index[*][?(@.name=='L4')].kind" \"struct\"
        // @has - "$.index[*][?(@.name=='L4')].inner.struct_type" \"unit\"
        pub struct L4;
    }
    // @has nested.json "$.index[*][?(@.inner.span=='l3::L4')].kind" \"import\"
    // @has - "$.index[*][?(@.inner.span=='l3::L4')].inner.glob" false
    pub use l3::L4;
}
