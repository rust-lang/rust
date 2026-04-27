//@ has "$.index[?(@.name=='redundant_at_def')]"
//@ has "$.index[?(@.name=='redundant_where')]"

//@ count "$.index[?(@.name=='redundant_at_def')].inner.function.generics.where_predicates[*]" 0
//@ has   "$.index[?(@.name=='redundant_at_def')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ count "$.index[?(@.name=='redundant_at_def')].inner.function.generics.params[0].kind.type.implied_bounds[*]" 0
pub fn redundant_at_def<T: Sized>(_: T) {}

//@ is   "$.index[?(@.name=='redundant_where')].inner.function.generics.where_predicates[0].bound_predicate.bounds[0].trait_bound.trait.path" '"Sized"'
//@ is   "$.index[?(@.name=='redundant_where')].inner.function.generics.where_predicates[0].bound_predicate.bounds[0].trait_bound.modifier" '"none"'
//@ count "$.index[?(@.name=='redundant_where')].inner.function.generics.params[0].kind.type.bounds[*]" 0
//@ count "$.index[?(@.name=='redundant_where')].inner.function.generics.params[0].kind.type.implied_bounds[*]" 0
pub fn redundant_where<T>(_: T)
where
    T: Sized,
{
}
