// ignore-tidy-linelength

#![feature(no_core)]
#![no_core]

// @count outlives.json "$.index[*][?(@.name=='foo')].inner.generics.params[*]" 3
// @is - "$.index[*][?(@.name=='foo')].inner.generics.params[0].name" \"\'a\"
// @is - "$.index[*][?(@.name=='foo')].inner.generics.params[1].name" \"\'b\"
// @is - "$.index[*][?(@.name=='foo')].inner.generics.params[2].name" '"T"'
// @is - "$.index[*][?(@.name=='foo')].inner.generics.params[2].kind.type.default" null
// @is - "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].kind" '"borrowed_ref"'
// @is - "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.lifetime" \"\'a\"
// @is - "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.mutable" false
// @is - "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.type.kind" '"borrowed_ref"'
// @is - "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.type.inner.lifetime" \"\'b\"
// @is - "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.type.inner.mutable" false
// @is - "$.index[*][?(@.name=='foo')].inner.decl.inputs[0][1].inner.type.inner.type" '{"inner": "T", "kind": "generic"}'
// @count - "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[*]" 2
// @is - "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].region_predicate.lifetime" \"\'b\"
// @count - "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].region_predicate.bounds[*]" 1
// @is - "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].region_predicate.bounds[0].outlives" \"\'a\"
// @is - "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[1].bound_predicate.type.inner" '"T"'
// @is - "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[1].bound_predicate.bounds[0].outlives" \"\'b\"
pub fn foo<'a, 'b: 'a, T: 'b>(_: &'a &'b T) {}
