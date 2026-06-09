#[repr(i32)]
//@ is "$.index[?(@.name=='Foo')].attrs[*].repr.int" '"i32"'
pub enum Foo {
    //@ is    "$.index[?(@.name=='Struct')].inner.variant.discriminant" null
    //@ count "$.index[?(@.name=='Struct')].inner.variant.kind.struct.fields[*]" 0
    Struct {},
    //@ is    "$.index[?(@.name=='StructWithDiscr')].inner.variant.discriminant" '{"expr": "42", "value": "42"}'
    //@ count "$.index[?(@.name=='StructWithDiscr')].inner.variant.kind.struct.fields[*]" 1
    StructWithDiscr { x: i32 } = 42,
    //@ is    "$.index[?(@.name=='StructWithHexDiscr')].inner.variant.discriminant"  '{"expr": "0x42", "value": "66"}'
    //@ count "$.index[?(@.name=='StructWithHexDiscr')].inner.variant.kind.struct.fields[*]" 2
    StructWithHexDiscr { x: i32, y: bool } = 0x42,
}
