// ignore-tidy-linelength

#![feature(no_core)]
#![no_core]

// @count "$.index[*][?(@.name=='foo')].inner.generics.params[*]" 3
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates" []
// @is "$.index[*][?(@.name=='foo')].inner.generics.params[0].name" \"\'a\"
// @is "$.index[*][?(@.name=='foo')].inner.generics.params[1].name" \"\'b\"
// @is "$.index[*][?(@.name=='foo')].inner.generics.params[2].name" '"T"'
// @is "$.index[*][?(@.name=='foo')].inner.generics.params[0].kind.lifetime.outlives" []
// @is "$.index[*][?(@.name=='foo')].inner.generics.params[1].kind.lifetime.outlives" [\"\'a\"]
// @is "$.index[*][?(@.name=='foo')].inner.generics.params[2].kind.type.default" null
// @count "$.index[*][?(@.name=='foo')].inner.generics.params[2].kind.type.bounds[*]" 1
// @is "$.index[*][?(@.name=='foo')].inner.generics.params[2].kind.type.bounds[0].outlives" \"\'b\"
// @is "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].kind" '"borrowed_ref"'
// @is "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.lifetime" \"\'a\"
// @is "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.mutable" false
// @is "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.type.kind" '"borrowed_ref"'
// @is "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.type.inner.lifetime" \"\'b\"
// @is "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.type.inner.mutable" false
// @is "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.type.inner.type" '{"inner": "T", "kind": "generic"}'
pub fn foo<'a, 'b: 'a, T: 'b>(_: &'a &'b T) {}
