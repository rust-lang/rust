#![feature(no_core)]
#![no_core]

pub enum Foo {
    // @is "$.index[*][?(@.name=='Unit')].inner.kind" '"unit"'
    // @set Unit = "$.index[*][?(@.name=='Unit')].id"
    // @is "$.index[*][?(@.name=='Unit')].inner.fields" []
    Unit,
    // @is "$.index[*][?(@.name=='Named')].inner.kind" '"named_fields"'
    // @set Named = "$.index[*][?(@.name=='Named')].id"
    // @is "$.index[*][?(@.name=='Named')].inner.fields" []
    Named {},
    // @is "$.index[*][?(@.name=='Tuple')].inner.kind" '"tuple"'
    // @set Tuple = "$.index[*][?(@.name=='Tuple')].id"
    // @is "$.index[*][?(@.name=='Tuple')].inner.fields" []
    Tuple(),
    // @is "$.index[*][?(@.name=='NamedField')].inner.kind" '"named_fields"'
    // @set NamedField = "$.index[*][?(@.name=='NamedField')].id"
    // @set x = "$.index[*][?(@.name=='x' && @.kind=='field')].id"
    // @is "$.index[*][?(@.name=='NamedField')].inner.fields[*]" $x
    NamedField { x: i32 },
    // @is "$.index[*][?(@.name=='TupleField')].inner.kind" '"tuple"'
    // @set TupleField = "$.index[*][?(@.name=='TupleField')].id"
    // @set tup_field = "$.index[*][?(@.name=='0' && @.kind=='field')].id"
    // @is "$.index[*][?(@.name=='TupleField')].inner.fields[*]" $tup_field
    TupleField(i32),
}

// @is    "$.index[*][?(@.name=='Foo')].inner.variants[0]" $Unit
// @is    "$.index[*][?(@.name=='Foo')].inner.variants[1]" $Named
// @is    "$.index[*][?(@.name=='Foo')].inner.variants[2]" $Tuple
// @is    "$.index[*][?(@.name=='Foo')].inner.variants[3]" $NamedField
// @is    "$.index[*][?(@.name=='Foo')].inner.variants[4]" $TupleField
// @count "$.index[*][?(@.name=='Foo')].inner.variants[*]" 5
