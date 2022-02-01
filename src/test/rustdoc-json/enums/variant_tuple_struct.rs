// @has variant_tuple_struct.json "$.index[*][?(@.name=='EnumTupleStruct')].visibility" \"public\"
// @has - "$.index[*][?(@.name=='EnumTupleStruct')].kind" \"enum\"
pub enum EnumTupleStruct {
    // @has - "$.index[*][?(@.name=='VariantA')].inner.variant_kind" \"tuple\"
    VariantA(
        // @set field_0 = - "$.index[*][?(@.name=='0')].id"
        // @has - "$.index[*][?(@.name=='0')].kind" \"struct_field\"
        u32,
        // @set field_1 = - "$.index[*][?(@.name=='1')].id"
        // @has - "$.index[*][?(@.name=='1')].kind" \"struct_field\"
        String,
    ),
}

// @has - "$.index[*][?(@.name=='VariantA')].inner.variant_inner[*]" $field_0
// @has - "$.index[*][?(@.name=='VariantA')].inner.variant_inner[*]" $field_1
