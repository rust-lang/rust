//@ is '$.index[?(@.name=="on_lifetimes")].inner.function.generics.where_predicates' '[{"lifetime_predicate": {"lifetime": "'\''all", "outlives": ["'\''a", "'\''b", "'\''c"]}}]'
pub fn on_lifetimes<'a, 'b, 'c, 'all>()
where
    'all: 'a + 'b + 'c,
{
}

//@ count '$.index[?(@.name=="on_trait")].inner.function.generics.params[*]' 2
//@ is    '$.index[?(@.name=="on_trait")].inner.function.generics.params[0].name' \"\'a\"
//@ is    '$.index[?(@.name=="on_trait")].inner.function.generics.params[0].kind.lifetime.outlives' []
//@ is    '$.index[?(@.name=="on_trait")].inner.function.generics.params[1].name' '"T"'
//@ is    '$.index[?(@.name=="on_trait")].inner.function.generics.params[1].kind.type.bounds' []
//@ is    '$.index[?(@.name=="on_trait")].inner.function.generics.params[1].kind.type.bounds' []
//@ count '$.index[?(@.name=="on_trait")].inner.function.generics.where_predicates[*]' 1
//@ is    '$.index[?(@.name=="on_trait")].inner.function.generics.where_predicates[0].bound_predicate.type.generic' '"T"'
//@ count '$.index[?(@.name=="on_trait")].inner.function.generics.where_predicates[0].bound_predicate.bounds[*]' 1
//@ is    '$.index[?(@.name=="on_trait")].inner.function.generics.where_predicates[0].bound_predicate.bounds[0].outlives' \"\'a\"
pub fn on_trait<'a, T>()
where
    T: 'a,
{
}
