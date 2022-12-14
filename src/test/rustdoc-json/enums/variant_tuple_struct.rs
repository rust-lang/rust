// @is "$.index[*][?(@.name=='EnumTupleStruct')].visibility" \"public\"
// @is "$.index[*][?(@.name=='EnumTupleStruct')].kind" \"enum\"
pub enum EnumTupleStruct {
    // @is "$.index[*][?(@.name=='VariantA')].inner.variant_kind" \"tuple\"
    // @is "$.index[*][?(@.name=='0')].kind" \"struct_field\"
    // @is "$.index[*][?(@.name=='1')].kind" \"struct_field\"
    VariantA(u32, String),
}
