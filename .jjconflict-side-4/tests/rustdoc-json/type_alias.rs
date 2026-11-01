//@ jq_set IntVec = '.index[] | select(.name == "IntVec").id'
//@ jq_is '.index[] | select(.name == "IntVec").visibility' '"public"'
//@ jq_is '.index[] | select(.name == "IntVec").inner | has("type_alias")' true
//@ jq_is '.index[] | select(.name == "IntVec").span.filename' $FILE
pub type IntVec = Vec<u32>;

//@ jq_is '.index[] | select(.name == "f").inner.function.sig.output.resolved_path.id' $IntVec
pub fn f() -> IntVec {
    vec![0; 32]
}

//@ !jq_is '.index[] | select(.name == "g").inner.function.sig.output.resolved_path.id' $IntVec
pub fn g() -> Vec<u32> {
    vec![0; 32]
}
