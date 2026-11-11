//@ is "$.index[?(@.name=='EnumTupleStruct')].visibility" \"public\"
//@ has "$.index[?(@.name=='EnumTupleStruct')].inner.enum"
pub enum EnumTupleStruct {
    //@ has "$.index[?(@.name=='0')].inner.struct_field"
    //@ set f0 = "$.index[?(@.name=='0')].id"
    //@ has "$.index[?(@.name=='1')].inner.struct_field"
    //@ set f1 = "$.index[?(@.name=='1')].id"
    //@ ismany "$.index[?(@.name=='VariantA')].inner.variant.kind.tuple[*]" $f0 $f1
    VariantA(u32, String),
}
