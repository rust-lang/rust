#[repr(u64)]
pub enum U64 {
    //@ arg u64min .index[] | select(.name == "U64Min").inner.variant.discriminant?
    //@ jq $u64min.value? == "0"
    //@ jq $u64min.expr? == "u64::MIN"
    U64Min = u64::MIN,
    //@ arg u64max .index[] | select(.name == "U64Max").inner.variant.discriminant?
    //@ jq $u64max.value? == "18446744073709551615"
    //@ jq $u64max.expr? == "u64::MAX"
    U64Max = u64::MAX,
}

#[repr(i64)]
pub enum I64 {
    //@ arg i64min .index[] | select(.name == "I64Min").inner.variant.discriminant?
    //@ jq $i64min.value? == "-9223372036854775808"
    //@ jq $i64min.expr? == "i64::MIN"
    I64Min = i64::MIN,
    //@ arg i64max .index[] | select(.name == "I64Max").inner.variant.discriminant?
    //@ jq $i64max.value? == "9223372036854775807"
    //@ jq $i64max.expr? == "i64::MAX"
    I64Max = i64::MAX,
}

#[repr(u128)]
pub enum U128 {
    //@ arg u128min .index[] | select(.name == "U128Min").inner.variant.discriminant?
    //@ jq $u128min.value? == "0"
    //@ jq $u128min.expr? == "u128::MIN"
    U128Min = u128::MIN,
    //@ arg u128max .index[] | select(.name == "U128Max").inner.variant.discriminant?
    //@ jq $u128max.value? == "340282366920938463463374607431768211455"
    //@ jq $u128max.expr? == "u128::MAX"
    U128Max = u128::MAX,
}

#[repr(i128)]
pub enum I128 {
    //@ arg i128min .index[] | select(.name == "I128Min").inner.variant.discriminant?
    //@ jq $i128min.value? == "-170141183460469231731687303715884105728"
    //@ jq $i128min.expr? == "i128::MIN"
    I128Min = i128::MIN,
    //@ arg i128max .index[] | select(.name == "I128Max").inner.variant.discriminant?
    //@ jq $i128max.value? == "170141183460469231731687303715884105727"
    //@ jq $i128max.expr? == "i128::MAX"
    I128Max = i128::MAX,
}
