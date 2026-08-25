#[repr(u32)]
pub enum Foo {
    //@ is "$.index[?(@.name=='Basic')].inner.variant.discriminant.value" '"0"'
    //@ is "$.index[?(@.name=='Basic')].inner.variant.discriminant.expr" '"0"'
    Basic = 0,
    //@ is "$.index[?(@.name=='Suffix')].inner.variant.discriminant.value" '"10"'
    //@ is "$.index[?(@.name=='Suffix')].inner.variant.discriminant.expr" '"10u32"'
    Suffix = 10u32,
    //@ is "$.index[?(@.name=='Underscore')].inner.variant.discriminant.value" '"100"'
    //@ is "$.index[?(@.name=='Underscore')].inner.variant.discriminant.expr" '"1_0_0"'
    Underscore = 1_0_0,
    //@ is "$.index[?(@.name=='SuffixUnderscore')].inner.variant.discriminant.value" '"1000"'
    //@ is "$.index[?(@.name=='SuffixUnderscore')].inner.variant.discriminant.expr" '"1_0_0_0u32"'
    SuffixUnderscore = 1_0_0_0u32,
}
