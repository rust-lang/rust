// @is "$.index[*][?(@.name=='EnumTupleStruct')].visibility" \"public\"
// @is "$.index[*][?(@.name=='EnumTupleStruct')].kind" \"enum\"
pub enum EnumTupleStruct {
    // @is "$.index[*][?(@.name=='0')].kind" \"struct_field\"
    // @set f0 = "$.index[*][?(@.name=='0')].id"
    // @is "$.index[*][?(@.name=='1')].kind" \"struct_field\"
    // @set f1 = "$.index[*][?(@.name=='1')].id"
    // @ismany "$.index[*][?(@.name=='VariantA')].inner.kind.tuple[*]" $f0 $f1
    VariantA(u32, String),
}
