// edition:2018

// @has nested.json "$.index.['0:0'].kind" \"module\"
// @has - "$.index.['0:0'].inner.is_crate" true
// @has - "$.index.['0:0'].inner.items[*]" \"0:3\"

// @has nested.json "$.index.['0:3'].kind" \"module\"
// @has - "$.index.['0:3'].inner.is_crate" false
// @has - "$.index.['0:3'].inner.items[*]" \"0:4\"
// @has - "$.index.['0:3'].inner.items[*]" \"0:7\"
pub mod l1 {

    // @has nested.json "$.index.['0:4'].kind" \"module\"
    // @has - "$.index.['0:4'].inner.is_crate" false
    // @has - "$.index.['0:4'].inner.items[*]" \"0:5\"
    pub mod l3 {

        // @has nested.json "$.index.['0:5'].kind" \"struct\"
        // @has - "$.index.['0:5'].inner.struct_type" \"unit\"
        pub struct L4;
    }
    // @has nested.json "$.index.['0:7'].kind" \"import\"
    // @has - "$.index.['0:7'].inner.glob" false
    pub use l3::L4;
}
