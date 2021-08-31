// @has variant_struct.json "$.index[*][?(@.name=='EnumStruct')].visibility" \"public\"
// @has - "$.index[*][?(@.name=='EnumStruct')].kind" \"enum\"
pub enum EnumStruct {
    // @has - "$.index[*][?(@.name=='VariantS')].inner.variant_kind" \"struct\"
    // @has - "$.index[*][?(@.name=='x')]"
    // @has - "$.index[*][?(@.name=='y')]"
    VariantS {
        x: u32,
        y: String,
    },
}
