pub enum Foo {
    //@ set Unit = "$.index[?(@.name=='Unit')].id"
    //@ is "$.index[?(@.name=='Unit')].inner.variant.kind" '"plain"'
    Unit,
    //@ set Named = "$.index[?(@.name=='Named')].id"
    //@ is "$.index[?(@.name=='Named')].inner.variant.kind.struct" '{"fields": [], "has_stripped_fields": false}'
    Named {},
    //@ set Tuple = "$.index[?(@.name=='Tuple')].id"
    //@ is "$.index[?(@.name=='Tuple')].inner.variant.kind.tuple" []
    Tuple(),
    //@ set NamedField = "$.index[?(@.name=='NamedField')].id"
    //@ set x = "$.index[?(@.name=='x' && @.inner.struct_field)].id"
    //@ is "$.index[?(@.name=='NamedField')].inner.variant.kind.struct.fields[*]" $x
    //@ is "$.index[?(@.name=='NamedField')].inner.variant.kind.struct.has_stripped_fields" false
    NamedField { x: i32 },
    //@ set TupleField = "$.index[?(@.name=='TupleField')].id"
    //@ set tup_field = "$.index[?(@.name=='0' && @.inner.struct_field)].id"
    //@ is "$.index[?(@.name=='TupleField')].inner.variant.kind.tuple[*]" $tup_field
    TupleField(i32),
}

//@ is    "$.index[?(@.name=='Foo')].inner.enum.variants[0]" $Unit
//@ is    "$.index[?(@.name=='Foo')].inner.enum.variants[1]" $Named
//@ is    "$.index[?(@.name=='Foo')].inner.enum.variants[2]" $Tuple
//@ is    "$.index[?(@.name=='Foo')].inner.enum.variants[3]" $NamedField
//@ is    "$.index[?(@.name=='Foo')].inner.enum.variants[4]" $TupleField
//@ count "$.index[?(@.name=='Foo')].inner.enum.variants[*]" 5
