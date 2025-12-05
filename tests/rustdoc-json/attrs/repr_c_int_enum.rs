//@ count "$.index[?(@.name=='Foo')].attrs" 1
//@ is "$.index[?(@.name=='Foo')].attrs[*].repr.kind" '"c"'
//@ is "$.index[?(@.name=='Foo')].attrs[*].repr.int" '"u8"'
//@ is "$.index[?(@.name=='Foo')].attrs[*].repr.packed" null
//@ is "$.index[?(@.name=='Foo')].attrs[*].repr.align" 16
#[repr(C, u8)]
#[repr(align(16))]
pub enum Foo {
    A(bool) = b'A',
    B(char) = b'C',
}
