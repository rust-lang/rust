// @has "$.index[*][?(@.name=='EnumTupleStruct')].visibility" \"public\"
// @has "$.index[*][?(@.name=='EnumTupleStruct')].kind" \"enum\"
pub enum EnumTupleStruct {
    // @has "$.index[*][?(@.name=='VariantA')].inner.variant_kind" \"tuple\"
    // @has "$.index[*][?(@.name=='0')].kind" \"struct_field\"
    // @has "$.index[*][?(@.name=='1')].kind" \"struct_field\"
    VariantA(u32, String),
}
