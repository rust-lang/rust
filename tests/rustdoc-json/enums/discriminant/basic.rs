#[repr(i8)]
pub enum Ordering {
    //@ is "$.index[?(@.name=='Less')].inner.variant.discriminant.expr" '"-1"'
    //@ is "$.index[?(@.name=='Less')].inner.variant.discriminant.value" '"-1"'
    Less = -1,
    //@ is "$.index[?(@.name=='Equal')].inner.variant.discriminant.expr" '"0"'
    //@ is "$.index[?(@.name=='Equal')].inner.variant.discriminant.value" '"0"'
    Equal = 0,
    //@ is "$.index[?(@.name=='Greater')].inner.variant.discriminant.expr" '"1"'
    //@ is "$.index[?(@.name=='Greater')].inner.variant.discriminant.value" '"1"'
    Greater = 1,
}
