#![no_std]

//@ is "$.index[?(@.name=='I8')].attrs[*].repr.int" '"i8"'
//@ is "$.index[?(@.name=='I8')].attrs[*].repr.kind" '"rust"'
//@ is "$.index[?(@.name=='I8')].attrs[*].repr.align" null
//@ is "$.index[?(@.name=='I8')].attrs[*].repr.packed" null
#[repr(i8)]
pub enum I8 {
    First,
}

//@ is "$.index[?(@.name=='I32')].attrs[*].repr.int" '"i32"'
//@ is "$.index[?(@.name=='I32')].attrs[*].repr.kind" '"rust"'
//@ is "$.index[?(@.name=='I32')].attrs[*].repr.align" null
//@ is "$.index[?(@.name=='I32')].attrs[*].repr.packed" null
#[repr(i32)]
pub enum I32 {
    First,
}

//@ is "$.index[?(@.name=='Usize')].attrs[*].repr.int" '"usize"'
//@ is "$.index[?(@.name=='Usize')].attrs[*].repr.kind" '"rust"'
//@ is "$.index[?(@.name=='Usize')].attrs[*].repr.align" null
//@ is "$.index[?(@.name=='Usize')].attrs[*].repr.packed" null
#[repr(usize)]
pub enum Usize {
    First,
}
