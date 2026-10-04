pub enum Foo {
    //@ is "$.index[?(@.name=='Has')].inner.variant.discriminant" '{"expr":"0", "value":"0"}'
    Has = 0,
    //@ is "$.index[?(@.name=='Doesnt')].inner.variant.discriminant" null
    Doesnt,
    //@ is "$.index[?(@.name=='AlsoDoesnt')].inner.variant.discriminant" null
    AlsoDoesnt,
    //@ is "$.index[?(@.name=='AlsoHas')].inner.variant.discriminant" '{"expr":"44", "value":"44"}'
    AlsoHas = 44,
}
