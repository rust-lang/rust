#![no_std]

//@ count "$.index[?(@.name=='ReprCStruct')].attrs" 1
//@ is "$.index[?(@.name=='ReprCStruct')].attrs[*].repr.kind" '"c"'
//@ is "$.index[?(@.name=='ReprCStruct')].attrs[*].repr.int" null
//@ is "$.index[?(@.name=='ReprCStruct')].attrs[*].repr.packed" null
//@ is "$.index[?(@.name=='ReprCStruct')].attrs[*].repr.align" null
#[repr(C)]
pub struct ReprCStruct(pub i64);

//@ count "$.index[?(@.name=='ReprCEnum')].attrs" 1
//@ is "$.index[?(@.name=='ReprCEnum')].attrs[*].repr.kind" '"c"'
//@ is "$.index[?(@.name=='ReprCEnum')].attrs[*].repr.int" null
//@ is "$.index[?(@.name=='ReprCEnum')].attrs[*].repr.packed" null
//@ is "$.index[?(@.name=='ReprCEnum')].attrs[*].repr.align" null
#[repr(C)]
pub enum ReprCEnum {
    First,
}

//@ count "$.index[?(@.name=='ReprCUnion')].attrs" 1
//@ is "$.index[?(@.name=='ReprCUnion')].attrs[*].repr.kind" '"c"'
//@ is "$.index[?(@.name=='ReprCUnion')].attrs[*].repr.int" null
//@ is "$.index[?(@.name=='ReprCUnion')].attrs[*].repr.packed" null
//@ is "$.index[?(@.name=='ReprCUnion')].attrs[*].repr.align" null
#[repr(C)]
pub union ReprCUnion {
    pub left: i64,
    pub right: u64,
}
