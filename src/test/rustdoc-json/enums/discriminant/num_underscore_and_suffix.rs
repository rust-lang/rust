#[repr(u32)]
pub enum Foo {
    // @is "$.index[*][?(@.name=='Basic')].inner.variant_inner.value" '"0"'
    // @is "$.index[*][?(@.name=='Basic')].inner.variant_inner.expr" '"0"'
    Basic = 0,
    // @is "$.index[*][?(@.name=='Suffix')].inner.variant_inner.value" '"10"'
    // @is "$.index[*][?(@.name=='Suffix')].inner.variant_inner.expr" '"10u32"'
    Suffix = 10u32,
    // @is "$.index[*][?(@.name=='Underscore')].inner.variant_inner.value" '"100"'
    // @is "$.index[*][?(@.name=='Underscore')].inner.variant_inner.expr" '"1_0_0"'
    Underscore = 1_0_0,
    // @is "$.index[*][?(@.name=='SuffixUnderscore')].inner.variant_inner.value" '"1000"'
    // @is "$.index[*][?(@.name=='SuffixUnderscore')].inner.variant_inner.expr" '"1_0_0_0u32"'
    SuffixUnderscore = 1_0_0_0u32,
}
