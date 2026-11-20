#[repr(i32)]
//@ jq_is '.index[] | select(.name == "Foo").attrs[0].repr.int' '"i32"'
pub enum Foo {
    //@ jq_set struct = '.index[] | select(.name == "Struct").inner.variant'
    //@ jq_is '$struct | has("discriminant")' 'true'
    //@ jq_is '$struct.discriminant?' 'null'
    //@ jq_count '$struct.kind?.struct.fields[]' 0
    Struct {},
    //@ jq_set struct_with_discr = '.index[] | select(.name == "StructWithDiscr").inner.variant'
    //@ jq_is '$struct_with_discr.discriminant?' '{"expr": "42", "value": "42"}'
    //@ jq_count '$struct_with_discr.kind?.struct.fields' 1
    StructWithDiscr { x: i32 } = 42,
    //@ jq_set struct_with_hex_discr = '.index[] | select(.name == "StructWithHexDiscr").inner.variant'
    //@ jq_is '$struct_with_hex_discr.discriminant?' '{"expr": "0x42", "value": "66"}'
    //@ jq_count '$struct_with_hex_discr.kind?.struct.fields[]' 2
    StructWithHexDiscr { x: i32, y: bool } = 0x42,
}
