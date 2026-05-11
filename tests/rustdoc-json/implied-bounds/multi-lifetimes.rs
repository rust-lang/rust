//@ is   "$.index[?(@.name=='Pair')].inner.struct.generics.params[2].name" '"T"'
//@ has  "$.index[?(@.name=='Pair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
//@ has  "$.index[?(@.name=='Pair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'b\")]"
//@ has  "$.index[?(@.name=='Pair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub struct Pair<'a, 'b, T>(&'a T, &'b T);

//@ is   "$.index[?(@.name=='require_pair')].inner.function.generics.params[2].name" '"T"'
//@ has  "$.index[?(@.name=='require_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
//@ has  "$.index[?(@.name=='require_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'b\")]"
//@ has  "$.index[?(@.name=='require_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub fn require_pair<'a, 'b, T>(_: &'a &'b T) -> Pair<'a, 'b, T> {
    todo!()
}

//@ is   "$.index[?(@.name=='OutlivePair')].inner.struct.generics.params[2].name" '"T"'
//@ has  "$.index[?(@.name=='OutlivePair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
//@ has  "$.index[?(@.name=='OutlivePair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'b\")]"
//@ has  "$.index[?(@.name=='OutlivePair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub struct OutlivePair<'a, 'b: 'a, T>(&'a T, &'b T);

//@ is   "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[1].name" \"\'b\"
//@ is   "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[2].name" '"T"'
// FIXME: Eventually we also want `implied_bounds` on lifetimes too. When implemented,
// enable the following test too:
// - @ is   "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[1].kind.lifetime.implied_bounds" '["\'a"]'
//@ has  "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
//@ has  "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'b\")]"
//@ has  "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub fn require_outlive_pair<'a, 'b, T>(x: &'a T, y: &'b T) -> OutlivePair<'a, 'b, T> {
    todo!()
}
