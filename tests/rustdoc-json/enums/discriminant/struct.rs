#[repr(i32)]
//@ jq .index[] | select(.name == "Foo").attrs == ["#[repr(i32)]"]
pub enum Foo {
    //@ arg struct .index[] | select(.name == "Struct").inner.variant
    //@ jq $struct.discriminant? == null
    //@ jq $struct.kind?.struct.fields? | length == 0
    Struct {},
    //@ arg struct_with_discr .index[] | select(.name == "StructWithDiscr").inner.variant
    //@ jq $struct_with_discr.discriminant? == {"expr": "42", "value": "42"}
    //@ jq $struct_with_discr.kind?.struct.fields? | length == 1
    StructWithDiscr { x: i32 } = 42,
    //@ arg struct_with_hex_discr .index[] | select(.name == "StructWithHexDiscr").inner.variant
    //@ jq $struct_with_hex_discr.discriminant? == {"expr": "0x42", "value": "66"}
    //@ jq $struct_with_hex_discr.kind?.struct.fields? | length == 2
    StructWithHexDiscr { x: i32, y: bool } = 0x42,
}
