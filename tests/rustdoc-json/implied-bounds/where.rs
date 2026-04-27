//@ count "$.index[?(@.name=='where_sized')].inner.function.generics.params[0].kind.type.bounds[*]" 0
//@ is   "$.index[?(@.name=='where_sized')].inner.function.generics.where_predicates[0].bound_predicate.bounds[0].trait_bound.trait.path" '"Sized"'
//@ !has "$.index[?(@.name=='where_sized')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn where_sized<T>(value: T)
where
    T: Sized,
{
    let _ = value;
}

//@ count "$.index[?(@.name=='where_unsized')].inner.function.generics.params[0].kind.type.implied_bounds[*]" 0
pub fn where_unsized<T>(value: &T)
where
    T: ?Sized,
{
    let _ = value;
}
