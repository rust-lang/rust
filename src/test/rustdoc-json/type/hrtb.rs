
// @has hrtb.json

// @is - "$.index[*][?(@.name=='genfn')].inner.generics.where_predicates[0].bound_predicate.type" '{"inner": "F","kind": "generic"}'
// @is - "$.index[*][?(@.name=='genfn')].inner.generics.where_predicates[0].bound_predicate.generic_params" '[{"kind": {"lifetime": {"outlives": []}},"name": "'\''a"},{"kind": {"lifetime": {"outlives": []}},"name": "'\''b"}]'
pub fn genfn<F>(f: F)
where
    for<'a, 'b> F: Fn(&'a i32, &'b i32),
{
    let zero = 0;
    f(&zero, &zero);
}

// @is - "$.index[*][?(@.name=='dynfn')].inner.generics" '{"params": [], "where_predicates": []}'
// @is - "$.index[*][?(@.name=='dynfn')].inner.generics" '{"params": [], "where_predicates": []}'
// @is - "$.index[*][?(@.name=='dynfn')].inner.decl.inputs[0][1].kind" '"borrowed_ref"'
// @is - "$.index[*][?(@.name=='dynfn')].inner.decl.inputs[0][1].kind" '"borrowed_ref"'
// @is - "$.index[*][?(@.name=='dynfn')].inner.decl.inputs[0][1].inner.type.kind" '"resolved_path"'
// @is - "$.index[*][?(@.name=='dynfn')].inner.decl.inputs[0][1].inner.type.inner.name" '"Fn"'
pub fn dynfn(f: &dyn for<'a, 'b> Fn(&'a i32, &'b i32)) {
    let zero = 0;
    f(&zero, &zero);
}
