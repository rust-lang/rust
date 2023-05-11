// ignore-tidy-linelength

#[repr(u32)]
// @is "$.index[*][?(@.name=='Foo')].attrs" '["#[repr(u32)]"]'
pub enum Foo {
    // @is    "$.index[*][?(@.name=='Tuple')].inner.discriminant" null
    // @count "$.index[*][?(@.name=='Tuple')].inner.kind.tuple[*]" 0
    Tuple(),
    // @is    "$.index[*][?(@.name=='TupleWithDiscr')].inner.discriminant" '{"expr": "1", "value": "1"}'
    // @count "$.index[*][?(@.name=='TupleWithDiscr')].inner.kind.tuple[*]" 1
    TupleWithDiscr(i32) = 1,
    // @is    "$.index[*][?(@.name=='TupleWithBinDiscr')].inner.discriminant" '{"expr": "0b10", "value": "2"}'
    // @count "$.index[*][?(@.name=='TupleWithBinDiscr')].inner.kind.tuple[*]" 2
    TupleWithBinDiscr(i32, i32) = 0b10,
}
