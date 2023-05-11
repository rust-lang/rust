// ignore-tidy-linelength

#![feature(no_core)]
#![no_core]

// @is "$.index[*][?(@.name=='longest')].inner.generics.params[0].name"  \"\'a\"
// @is "$.index[*][?(@.name=='longest')].inner.generics.params[0].kind"  '{"lifetime": {"outlives": []}}'
// @is "$.index[*][?(@.name=='longest')].inner.generics.params[0].kind"  '{"lifetime": {"outlives": []}}'
// @count "$.index[*][?(@.name=='longest')].inner.generics.params[*]" 1
// @is "$.index[*][?(@.name=='longest')].inner.generics.where_predicates" []

// @count "$.index[*][?(@.name=='longest')].inner.decl.inputs[*]" 2
// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[0][0]" '"l"'
// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[1][0]" '"r"'

// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[0][1].kind" '"borrowed_ref"'
// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[0][1].inner.lifetime" \"\'a\"
// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[0][1].inner.mutable" false
// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[0][1].inner.type" '{"inner": "str", "kind": "primitive"}'

// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[1][1].kind" '"borrowed_ref"'
// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[1][1].inner.lifetime" \"\'a\"
// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[1][1].inner.mutable" false
// @is "$.index[*][?(@.name=='longest')].inner.decl.inputs[1][1].inner.type" '{"inner": "str", "kind": "primitive"}'

// @is "$.index[*][?(@.name=='longest')].inner.decl.output.kind" '"borrowed_ref"'
// @is "$.index[*][?(@.name=='longest')].inner.decl.output.inner.lifetime" \"\'a\"
// @is "$.index[*][?(@.name=='longest')].inner.decl.output.inner.mutable" false
// @is "$.index[*][?(@.name=='longest')].inner.decl.output.inner.type" '{"inner": "str", "kind": "primitive"}'

pub fn longest<'a>(l: &'a str, r: &'a str) -> &'a str {
    if l.len() > r.len() { l } else { r }
}
