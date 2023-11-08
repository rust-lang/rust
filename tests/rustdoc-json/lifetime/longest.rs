// ignore-tidy-linelength

// @is "$.index[*][?(@.name=='longest')].inner.function.generics.params[0].name"  \"\'a\"
// @is "$.index[*][?(@.name=='longest')].inner.function.generics.params[0].kind"  '{"lifetime": {"outlives": []}}'
// @is "$.index[*][?(@.name=='longest')].inner.function.generics.params[0].kind"  '{"lifetime": {"outlives": []}}'
// @count "$.index[*][?(@.name=='longest')].inner.function.generics.params[*]" 1
// @is "$.index[*][?(@.name=='longest')].inner.function.generics.where_predicates" []

// @count "$.index[*][?(@.name=='longest')].inner.function.decl.inputs[*]" 2
// @is "$.index[*][?(@.name=='longest')].inner.function.decl.inputs[0][0]" '"l"'
// @is "$.index[*][?(@.name=='longest')].inner.function.decl.inputs[1][0]" '"r"'

// @is "$.index[*][?(@.name=='longest')].inner.function.decl.inputs[0][1].borrowed_ref.lifetime" \"\'a\"
// @is "$.index[*][?(@.name=='longest')].inner.function.decl.inputs[0][1].borrowed_ref.mutable" false
// @is "$.index[*][?(@.name=='longest')].inner.function.decl.inputs[0][1].borrowed_ref.type.primitive" \"str\"

// @is "$.index[*][?(@.name=='longest')].inner.function.decl.inputs[1][1].borrowed_ref.lifetime" \"\'a\"
// @is "$.index[*][?(@.name=='longest')].inner.function.decl.inputs[1][1].borrowed_ref.mutable" false
// @is "$.index[*][?(@.name=='longest')].inner.function.decl.inputs[1][1].borrowed_ref.type.primitive" \"str\"

// @is "$.index[*][?(@.name=='longest')].inner.function.decl.output.borrowed_ref.lifetime" \"\'a\"
// @is "$.index[*][?(@.name=='longest')].inner.function.decl.output.borrowed_ref.mutable" false
// @is "$.index[*][?(@.name=='longest')].inner.function.decl.output.borrowed_ref.type.primitive" \"str\"

pub fn longest<'a>(l: &'a str, r: &'a str) -> &'a str {
    if l.len() > r.len() {
        l
    } else {
        r
    }
}
