#[repr(u64)]
pub enum U64 {
    //@ is "$.index[?(@.name=='U64Min')].inner.variant.discriminant.value" '"0"'
    //@ is "$.index[?(@.name=='U64Min')].inner.variant.discriminant.expr" '"u64::MIN"'
    U64Min = u64::MIN,
    //@ is "$.index[?(@.name=='U64Max')].inner.variant.discriminant.value" '"18446744073709551615"'
    //@ is "$.index[?(@.name=='U64Max')].inner.variant.discriminant.expr" '"u64::MAX"'
    U64Max = u64::MAX,
}

#[repr(i64)]
pub enum I64 {
    //@ is "$.index[?(@.name=='I64Min')].inner.variant.discriminant.value" '"-9223372036854775808"'
    //@ is "$.index[?(@.name=='I64Min')].inner.variant.discriminant.expr" '"i64::MIN"'
    I64Min = i64::MIN,
    //@ is "$.index[?(@.name=='I64Max')].inner.variant.discriminant.value" '"9223372036854775807"'
    //@ is "$.index[?(@.name=='I64Max')].inner.variant.discriminant.expr" '"i64::MAX"'
    I64Max = i64::MAX,
}

#[repr(u128)]
pub enum U128 {
    //@ is "$.index[?(@.name=='U128Min')].inner.variant.discriminant.value" '"0"'
    //@ is "$.index[?(@.name=='U128Min')].inner.variant.discriminant.expr" '"u128::MIN"'
    U128Min = u128::MIN,
    //@ is "$.index[?(@.name=='U128Max')].inner.variant.discriminant.value" '"340282366920938463463374607431768211455"'
    //@ is "$.index[?(@.name=='U128Max')].inner.variant.discriminant.expr" '"u128::MAX"'
    U128Max = u128::MAX,
}

#[repr(i128)]
pub enum I128 {
    //@ is "$.index[?(@.name=='I128Min')].inner.variant.discriminant.value" '"-170141183460469231731687303715884105728"'
    //@ is "$.index[?(@.name=='I128Min')].inner.variant.discriminant.expr" '"i128::MIN"'
    I128Min = i128::MIN,
    //@ is "$.index[?(@.name=='I128Max')].inner.variant.discriminant.value" '"170141183460469231731687303715884105727"'
    //@ is "$.index[?(@.name=='I128Max')].inner.variant.discriminant.expr" '"i128::MAX"'
    I128Max = i128::MAX,
}
