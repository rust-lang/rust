#![no_std]

// Combinations of `#[repr(..)]` attributes.

//@ is "$.index[?(@.name=='ReprCPacked')].attrs" '[{"repr":{"align":null,"int":null,"kind":"c","packed":1}}]'
#[repr(C, packed)]
pub struct ReprCPacked {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='SeparateReprCPacked')].attrs" '[{"repr":{"align":null,"int":null,"kind":"c","packed":2}}]'
#[repr(C)]
#[repr(packed(2))]
pub struct SeparateReprCPacked {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='ReversedReprCPacked')].attrs" '[{"repr":{"align":null,"int":null,"kind":"c","packed":2}}]'
#[repr(packed(2), C)]
pub struct ReversedReprCPacked {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='ReprCAlign')].attrs" '[{"repr":{"align":16,"int":null,"kind":"c","packed":null}}]'
#[repr(C, align(16))]
pub struct ReprCAlign {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='SeparateReprCAlign')].attrs" '[{"repr":{"align":2,"int":null,"kind":"c","packed":null}}]'
#[repr(C)]
#[repr(align(2))]
pub struct SeparateReprCAlign {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='ReversedReprCAlign')].attrs" '[{"repr":{"align":2,"int":null,"kind":"c","packed":null}}]'
#[repr(align(2), C)]
pub struct ReversedReprCAlign {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='Transparent')].attrs" '[{"repr":{"align":null,"int":null,"kind":"transparent","packed":null}}]'
#[repr(transparent)]
pub struct Transparent(i64);
