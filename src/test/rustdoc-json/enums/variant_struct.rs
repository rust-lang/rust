// @is "$.index[*][?(@.name=='EnumStruct')].visibility" \"public\"
// @is "$.index[*][?(@.name=='EnumStruct')].kind" \"enum\"
pub enum EnumStruct {
    // @is "$.index[*][?(@.name=='VariantS')].inner.kind" \"named_fields\"
    // @is "$.index[*][?(@.name=='x')].kind" \"field\"
    // @is "$.index[*][?(@.name=='y')].kind" \"field\"
    VariantS { x: u32, y: String },
}
