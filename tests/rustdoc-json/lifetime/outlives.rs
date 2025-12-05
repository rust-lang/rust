//@ count "$.index[?(@.name=='foo')].inner.function.generics.params[*]" 3
//@ is "$.index[?(@.name=='foo')].inner.function.generics.where_predicates" []
//@ is "$.index[?(@.name=='foo')].inner.function.generics.params[0].name" \"\'a\"
//@ is "$.index[?(@.name=='foo')].inner.function.generics.params[1].name" \"\'b\"
//@ is "$.index[?(@.name=='foo')].inner.function.generics.params[2].name" '"T"'
//@ is "$.index[?(@.name=='foo')].inner.function.generics.params[0].kind.lifetime.outlives" []
//@ is "$.index[?(@.name=='foo')].inner.function.generics.params[1].kind.lifetime.outlives" [\"\'a\"]
//@ is "$.index[?(@.name=='foo')].inner.function.generics.params[2].kind.type.default" null
//@ count "$.index[?(@.name=='foo')].inner.function.generics.params[2].kind.type.bounds[*]" 1
//@ is "$.index[?(@.name=='foo')].inner.function.generics.params[2].kind.type.bounds[0].outlives" \"\'b\"
//@ is "$.index[?(@.name=='foo')].inner.function.sig.inputs[0][1].borrowed_ref.lifetime" \"\'a\"
//@ is "$.index[?(@.name=='foo')].inner.function.sig.inputs[0][1].borrowed_ref.is_mutable" false
//@ is "$.index[?(@.name=='foo')].inner.function.sig.inputs[0][1].borrowed_ref.type.borrowed_ref.lifetime" \"\'b\"
//@ is "$.index[?(@.name=='foo')].inner.function.sig.inputs[0][1].borrowed_ref.type.borrowed_ref.is_mutable" false
//@ is "$.index[?(@.name=='foo')].inner.function.sig.inputs[0][1].borrowed_ref.type.borrowed_ref.type.generic" \"T\"
pub fn foo<'a, 'b: 'a, T: 'b>(_: &'a &'b T) {}
