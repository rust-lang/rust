pub enum Foo {
    // @is "$.index[*][?(@.name=='Has')].inner.variant_inner" '{"expr":"0", "value":"0"}'
    Has = 0,
    // @is "$.index[*][?(@.name=='Doesnt')].inner.variant_inner" null
    Doesnt,
    // @is "$.index[*][?(@.name=='AlsoDoesnt')].inner.variant_inner" null
    AlsoDoesnt,
    // @is "$.index[*][?(@.name=='AlsoHas')].inner.variant_inner" '{"expr":"44", "value":"44"}'
    AlsoHas = 44,
}
