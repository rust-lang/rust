//@ is "$.index[?(@.name=='EnumStruct')].visibility" \"public\"
//@ has "$.index[?(@.name=='EnumStruct')].inner.enum"
pub enum EnumStruct {
    //@ has "$.index[?(@.name=='x')].inner.struct_field"
    //@ set x = "$.index[?(@.name=='x')].id"
    //@ has "$.index[?(@.name=='y')].inner.struct_field"
    //@ set y = "$.index[?(@.name=='y')].id"
    //@ ismany "$.index[?(@.name=='VariantS')].inner.variant.kind.struct.fields[*]" $x $y
    VariantS { x: u32, y: String },
}
