//@ is "$.index[?(@.name=='longest')].inner.function.generics.params[0].name"  \"\'a\"
//@ is "$.index[?(@.name=='longest')].inner.function.generics.params[0].kind"  '{"lifetime": {"outlives": []}}'
//@ is "$.index[?(@.name=='longest')].inner.function.generics.params[0].kind"  '{"lifetime": {"outlives": []}}'
//@ count "$.index[?(@.name=='longest')].inner.function.generics.params[*]" 1
//@ is "$.index[?(@.name=='longest')].inner.function.generics.where_predicates" []

//@ count "$.index[?(@.name=='longest')].inner.function.sig.inputs[*]" 2
//@ is "$.index[?(@.name=='longest')].inner.function.sig.inputs[0][0]" '"l"'
//@ is "$.index[?(@.name=='longest')].inner.function.sig.inputs[1][0]" '"r"'

//@ is "$.index[?(@.name=='longest')].inner.function.sig.inputs[0][1].borrowed_ref.lifetime" \"\'a\"
//@ is "$.index[?(@.name=='longest')].inner.function.sig.inputs[0][1].borrowed_ref.is_mutable" false
//@ is "$.index[?(@.name=='longest')].inner.function.sig.inputs[0][1].borrowed_ref.type.primitive" \"str\"

//@ is "$.index[?(@.name=='longest')].inner.function.sig.inputs[1][1].borrowed_ref.lifetime" \"\'a\"
//@ is "$.index[?(@.name=='longest')].inner.function.sig.inputs[1][1].borrowed_ref.is_mutable" false
//@ is "$.index[?(@.name=='longest')].inner.function.sig.inputs[1][1].borrowed_ref.type.primitive" \"str\"

//@ is "$.index[?(@.name=='longest')].inner.function.sig.output.borrowed_ref.lifetime" \"\'a\"
//@ is "$.index[?(@.name=='longest')].inner.function.sig.output.borrowed_ref.is_mutable" false
//@ is "$.index[?(@.name=='longest')].inner.function.sig.output.borrowed_ref.type.primitive" \"str\"

pub fn longest<'a>(l: &'a str, r: &'a str) -> &'a str {
    if l.len() > r.len() { l } else { r }
}
