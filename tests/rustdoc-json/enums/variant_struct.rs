// @is "$.index[*][?(@.name=='EnumStruct')].visibility" \"public\"
// @is "$.index[*][?(@.name=='EnumStruct')].kind" \"enum\"
pub enum EnumStruct {
    // @is "$.index[*][?(@.name=='x')].kind" \"struct_field\"
    // @set x = "$.index[*][?(@.name=='x')].id"
    // @is "$.index[*][?(@.name=='y')].kind" \"struct_field\"
    // @set y = "$.index[*][?(@.name=='y')].id"
    // @ismany "$.index[*][?(@.name=='VariantS')].inner.kind.struct.fields[*]" $x $y
    VariantS { x: u32, y: String },
}
