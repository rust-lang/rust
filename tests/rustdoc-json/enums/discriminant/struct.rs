// ignore-tidy-linelength

#[repr(i32)]
// @is "$.index[*][?(@.name=='Foo')].attrs" '["#[repr(i32)]"]'
pub enum Foo {
    // @is    "$.index[*][?(@.name=='Struct')].inner.discriminant" null
    // @count "$.index[*][?(@.name=='Struct')].inner.kind.struct.fields[*]" 0
    Struct {},
    // @is    "$.index[*][?(@.name=='StructWithDiscr')].inner.discriminant" '{"expr": "42", "value": "42"}'
    // @count "$.index[*][?(@.name=='StructWithDiscr')].inner.kind.struct.fields[*]" 1
    StructWithDiscr { x: i32 } = 42,
    // @is    "$.index[*][?(@.name=='StructWithHexDiscr')].inner.discriminant"  '{"expr": "0x42", "value": "66"}'
    // @count "$.index[*][?(@.name=='StructWithHexDiscr')].inner.kind.struct.fields[*]" 2
    StructWithHexDiscr { x: i32, y: bool } = 0x42,
}
