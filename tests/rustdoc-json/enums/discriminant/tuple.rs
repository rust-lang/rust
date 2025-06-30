#[repr(u32)]
//@ jq .index[] | select(.name == "Foo").attrs == ["#[repr(u32)]"]
pub enum Foo {
    //@ arg tuple .index[] | select(.name == "Tuple").inner.variant
    //@ jq $tuple.discriminant? == null
    //@ jq $tuple.kind?.tuple | length == 0
    Tuple(),
    //@ arg tuple_with_discr .index[] | select(.name == "TupleWithDiscr").inner.variant
    //@ jq $tuple_with_discr.discriminant? == {"expr": "1", "value": "1"}
    //@ jq $tuple_with_discr.kind?.tuple | length == 1
    TupleWithDiscr(i32) = 1,
    //@ arg tuple_with_bin_discr .index[] | select(.name == "TupleWithBinDiscr").inner.variant
    //@ jq $tuple_with_bin_discr.discriminant? == {"expr": "0b10", "value": "2"}
    //@ jq $tuple_with_bin_discr.kind?.tuple | length == 2
    TupleWithBinDiscr(i32, i32) = 0b10,
}
