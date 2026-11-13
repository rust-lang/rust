//@ is "$.index[?(@.name=='genfn')].inner.function.generics.where_predicates[0].bound_predicate.type" '{"generic": "F"}'
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
//@ is "$.index[?(@.name=='dynfn')].inner.function.sig.inputs[0][1].borrowed_ref.type.dyn_trait.lifetime" null
//@ count "$.index[?(@.name=='dynfn')].inner.function.sig.inputs[0][1].borrowed_ref.type.dyn_trait.traits[*]" 1
//@ is "$.index[?(@.name=='dynfn')].inner.function.sig.inputs[0][1].borrowed_ref.type.dyn_trait.traits[0].generic_params" '[{"kind": {"lifetime": {"outlives": []}},"name": "'\''a"},{"kind": {"lifetime": {"outlives": []}},"name": "'\''b"}]'
//@ is "$.index[?(@.name=='dynfn')].inner.function.sig.inputs[0][1].borrowed_ref.type.dyn_trait.traits[0].trait.path" '"Fn"'
pub fn dynfn(f: &dyn for<'a, 'b> Fn(&'a i32, &'b i32)) {
    let zero = 0;
    f(&zero, &zero);
}
