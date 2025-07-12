//@ arg enum_struct .index[] | select(.name == "EnumStruct")
//@ jq $enum_struct.visibility == "public"
//@ jq $enum_struct.inner.enum
pub enum EnumStruct {
    //@ arg x .index[] | select(.name == "x")
    //@ arg y .index[] | select(.name == "y")
    //@ jq [[$x, $y][].inner.struct_field] | all
    //@ jq .index[] | select(.name == "VariantS").inner.variant.kind?.struct.fields? == [$x.id, $y.id]
    VariantS { x: u32, y: String },
}
