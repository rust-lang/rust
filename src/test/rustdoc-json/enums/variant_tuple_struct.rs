// @is "$.index[*][?(@.name=='EnumTupleStruct')].visibility" \"public\"
// @is "$.index[*][?(@.name=='EnumTupleStruct')].kind" \"enum\"
pub enum EnumTupleStruct {
    // @is "$.index[*][?(@.name=='VariantA')].inner.kind" \"tuple\"
    // @is "$.index[*][?(@.name=='0')].kind" \"field\"
    // @is "$.index[*][?(@.name=='1')].kind" \"field\"
    VariantA(u32, String),
}
