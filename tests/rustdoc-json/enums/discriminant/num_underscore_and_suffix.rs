#[repr(u32)]
pub enum Foo {
    // @is "$.index[*][?(@.name=='Basic')].inner.discriminant.value" '"0"'
    // @is "$.index[*][?(@.name=='Basic')].inner.discriminant.expr" '"0"'
    Basic = 0,
    // @is "$.index[*][?(@.name=='Suffix')].inner.discriminant.value" '"10"'
    // @is "$.index[*][?(@.name=='Suffix')].inner.discriminant.expr" '"10u32"'
    Suffix = 10u32,
    // @is "$.index[*][?(@.name=='Underscore')].inner.discriminant.value" '"100"'
    // @is "$.index[*][?(@.name=='Underscore')].inner.discriminant.expr" '"1_0_0"'
    Underscore = 1_0_0,
    // @is "$.index[*][?(@.name=='SuffixUnderscore')].inner.discriminant.value" '"1000"'
    // @is "$.index[*][?(@.name=='SuffixUnderscore')].inner.discriminant.expr" '"1_0_0_0u32"'
    SuffixUnderscore = 1_0_0_0u32,
}
