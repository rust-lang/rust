#[repr(i8)]
pub enum Ordering {
    // @is "$.index[*][?(@.name=='Less')].inner.variant_inner.expr" '"-1"'
    // @is "$.index[*][?(@.name=='Less')].inner.variant_inner.value" '"-1"'
    Less = -1,
    // @is "$.index[*][?(@.name=='Equal')].inner.variant_inner.expr" '"0"'
    // @is "$.index[*][?(@.name=='Equal')].inner.variant_inner.value" '"0"'
    Equal = 0,
    // @is "$.index[*][?(@.name=='Greater')].inner.variant_inner.expr" '"1"'
    // @is "$.index[*][?(@.name=='Greater')].inner.variant_inner.value" '"1"'
    Greater = 1,
}
