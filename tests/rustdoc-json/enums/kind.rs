//@ arg foo .index[] | select(.name == "Foo").inner.enum.variants?
//@ jq $foo | length == 5

pub enum Foo {
    //@ arg unit .index[] | select(.name == "Unit")
    //@ jq $unit.id == $foo[0]
    //@ jq $unit.inner.variant.kind? == "plain"
    Unit,
    //@ arg named .index[] | select(.name == "Named")
    //@ jq $named.id == $foo[1]
    //@ jq $named.inner.variant.kind?.struct == {"fields": [], "has_stripped_fields": false}
    Named {},
    //@ arg tuple .index[] | select(.name == "Tuple")
    //@ jq $tuple.id == $foo[2]
    //@ jq $tuple.inner.variant.kind?.tuple == []
    Tuple(),
    //@ arg named_field .index[] | select(.name == "NamedField")
    //@ jq $named_field.id == $foo[3]
    //@ jq $named_field.inner.variant.kind?.struct.fields[]? == (.index[] | select(.name == "x" and .inner.struct_field).id)
    //@ jq $named_field.inner.variant.kind?.struct.has_stripped_fields? == false
    NamedField { x: i32 },
    //@ arg tuple_field .index[] | select(.name == "TupleField")
    //@ jq $tuple_field.id == $foo[4]
    //@ jq $tuple_field.inner.variant.kind?.tuple[]? == (.index[] | select(.name == "0" and .inner.struct_field).id)
    TupleField(i32),
}
