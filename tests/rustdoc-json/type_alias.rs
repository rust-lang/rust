//@ set IntVec = "$.index[?(@.name=='IntVec')].id"
//@ is "$.index[?(@.name=='IntVec')].visibility" \"public\"
//@ has "$.index[?(@.name=='IntVec')].inner.type_alias"
//@ is "$.index[?(@.name=='IntVec')].span.filename" $FILE
pub type IntVec = Vec<u32>;

//@ is "$.index[?(@.name=='f')].inner.function.sig.output.resolved_path.id" $IntVec
pub fn f() -> IntVec {
    vec![0; 32]
}

//@ !is "$.index[?(@.name=='g')].inner.function.sig.output.resolved_path.id" $IntVec
pub fn g() -> Vec<u32> {
    vec![0; 32]
}
