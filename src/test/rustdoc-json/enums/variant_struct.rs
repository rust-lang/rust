// @has "$.index[*][?(@.name=='EnumStruct')].visibility" \"public\"
// @has "$.index[*][?(@.name=='EnumStruct')].kind" \"enum\"
pub enum EnumStruct {
    // @has "$.index[*][?(@.name=='VariantS')].inner.variant_kind" \"struct\"
    // @has "$.index[*][?(@.name=='x')].kind" \"struct_field\"
    // @has "$.index[*][?(@.name=='y')].kind" \"struct_field\"
    VariantS {
        x: u32,
        y: String,
    },
}
