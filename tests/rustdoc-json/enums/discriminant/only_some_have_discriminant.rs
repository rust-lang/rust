pub enum Foo {
    // @is "$.index[*][?(@.name=='Has')].inner.discriminant" '{"expr":"0", "value":"0"}'
    Has = 0,
    // @is "$.index[*][?(@.name=='Doesnt')].inner.discriminant" null
    Doesnt,
    // @is "$.index[*][?(@.name=='AlsoDoesnt')].inner.discriminant" null
    AlsoDoesnt,
    // @is "$.index[*][?(@.name=='AlsoHas')].inner.discriminant" '{"expr":"44", "value":"44"}'
    AlsoHas = 44,
}
