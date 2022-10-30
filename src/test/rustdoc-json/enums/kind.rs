// ignore-tidy-linelength

#![feature(no_core)]
#![no_core]

pub enum Foo {
    // @set Unit = "$.index[*][?(@.name=='Unit')].id"
    // @is "$.index[*][?(@.name=='Unit')].inner.variant_kind" '"plain"'
    // @is "$.index[*][?(@.name=='Unit')].inner.variant_inner" null
    Unit,
    // @set Named = "$.index[*][?(@.name=='Named')].id"
    // @is "$.index[*][?(@.name=='Named')].inner.variant_kind" '"struct"'
    // @is "$.index[*][?(@.name=='Named')].inner.variant_inner" '{"fields": [], "fields_stripped": false}'
    Named {},
    // @set Tuple = "$.index[*][?(@.name=='Tuple')].id"
    // @is "$.index[*][?(@.name=='Tuple')].inner.variant_kind" '"tuple"'
    // @is "$.index[*][?(@.name=='Tuple')].inner.variant_inner" []
    Tuple(),
    // @set NamedField = "$.index[*][?(@.name=='NamedField')].id"
    // @set x = "$.index[*][?(@.name=='x' && @.kind=='struct_field')].id"
    // @is "$.index[*][?(@.name=='NamedField')].inner.variant_kind" '"struct"'
    // @is "$.index[*][?(@.name=='NamedField')].inner.variant_inner.fields[*]" $x
    // @is "$.index[*][?(@.name=='NamedField')].inner.variant_inner.fields_stripped" false
    NamedField { x: i32 },
    // @set TupleField = "$.index[*][?(@.name=='TupleField')].id"
    // @is "$.index[*][?(@.name=='TupleField')].inner.variant_kind" '"tuple"'
    // @set tup_field = "$.index[*][?(@.name=='0' && @.kind=='struct_field')].id"
    // @is "$.index[*][?(@.name=='TupleField')].inner.variant_inner[*]" $tup_field
    TupleField(i32),
}

// @is    "$.index[*][?(@.name=='Foo')].inner.variants[0]" $Unit
// @is    "$.index[*][?(@.name=='Foo')].inner.variants[1]" $Named
// @is    "$.index[*][?(@.name=='Foo')].inner.variants[2]" $Tuple
// @is    "$.index[*][?(@.name=='Foo')].inner.variants[3]" $NamedField
// @is    "$.index[*][?(@.name=='Foo')].inner.variants[4]" $TupleField
// @count "$.index[*][?(@.name=='Foo')].inner.variants[*]" 5
