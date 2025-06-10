//@ is "$.index[?(@.name=='genfn')].inner.function.generics.where_predicates[0].bound_predicate.type" 0
//@ is "$.types[0].generic" '"F"'
//@ is "$.index[?(@.name=='genfn')].inner.function.generics.where_predicates[0].bound_predicate.generic_params" '[{"kind": {"lifetime": {"outlives": []}},"name": "'\''a"},{"kind": {"lifetime": {"outlives": []}},"name": "'\''b"}]'
pub fn genfn<F>(f: F)
where
    for<'a, 'b> F: Fn(&'a i32, &'b i32),
{
    let zero = 0;
    f(&zero, &zero);
}

//@ is "$.index[?(@.name=='dynfn')].inner.function.generics" '{"params": [], "where_predicates": []}'
//@ is "$.index[?(@.name=='dynfn')].inner.function.generics" '{"params": [], "where_predicates": []}'
//@ is "$.index[?(@.name=='dynfn')].inner.function.sig.inputs[0][1]" 5
//@ is "$.types[5].borrowed_ref.type" 4
//@ is "$.types[4].dyn_trait.lifetime" null
//@ count "$.types[4].dyn_trait.traits[*]" 1
//@ is "$.types[4].dyn_trait.traits[0].generic_params" '[{"kind": {"lifetime": {"outlives": []}},"name": "'\''a"},{"kind": {"lifetime": {"outlives": []}},"name": "'\''b"}]'
//@ is "$.types[4].dyn_trait.traits[0].trait.path" '"Fn"'
pub fn dynfn(f: &dyn for<'a, 'b> Fn(&'a i32, &'b i32)) {
    let zero = 0;
    f(&zero, &zero);
}
