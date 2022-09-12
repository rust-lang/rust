// @is "$.index[*][?(@.name=='EnumStruct')].visibility" \"public\"
// @is "$.index[*][?(@.name=='EnumStruct')].kind" \"enum\"
pub enum EnumStruct {
    // @is "$.index[*][?(@.name=='VariantS')].inner.variant_kind" \"struct\"
    // @is "$.index[*][?(@.name=='x')].kind" \"struct_field\"
    // @is "$.index[*][?(@.name=='y')].kind" \"struct_field\"
    VariantS {
        x: u32,
        y: String,
    },
}
