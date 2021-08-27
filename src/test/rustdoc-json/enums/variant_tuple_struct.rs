// @has variant_tuple_struct.json "$.index[*][?(@.name=='EnumTupleStruct')].visibility" \"public\"
// @has - "$.index[*][?(@.name=='EnumTupleStruct')].kind" \"enum\"
pub enum EnumTupleStruct {
    // @has - "$.index[*][?(@.name=='VariantA')].inner.variant_kind" \"tuple\"
    VariantA(u32, String),
}
