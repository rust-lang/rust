#[repr(i8)]
pub enum Ordering {
    // @is "$.index[*][?(@.name=='Less')].inner.discriminant.expr" '"-1"'
    // @is "$.index[*][?(@.name=='Less')].inner.discriminant.value" '"-1"'
    Less = -1,
    // @is "$.index[*][?(@.name=='Equal')].inner.discriminant.expr" '"0"'
    // @is "$.index[*][?(@.name=='Equal')].inner.discriminant.value" '"0"'
    Equal = 0,
    // @is "$.index[*][?(@.name=='Greater')].inner.discriminant.expr" '"1"'
    // @is "$.index[*][?(@.name=='Greater')].inner.discriminant.value" '"1"'
    Greater = 1,
}
