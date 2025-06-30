//@ arg enum_tuple_struct .index[] | select(.name == "EnumTupleStruct")
//@ jq $enum_tuple_struct.visibility == "public"
//@ jq $enum_tuple_struct.inner.enum
pub enum EnumTupleStruct {
    //@ arg f0 .index[] | select(.name == "0")
    //@ arg f1 .index[] | select(.name == "1")
    //@ jq [[$f0, $f1][].inner.struct_field] | all
    //@ jq .index[] | select(.name == "VariantA").inner.variant.kind?.tuple == [$f0.id, $f1.id]
    VariantA(u32, String),
}
