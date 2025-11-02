//@ is   "$.index[?(@.name=='Pair')].inner.struct.generics.params[2].name" '"T"'
//@ has  "$.index[?(@.name=='Pair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
//@ has  "$.index[?(@.name=='Pair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'b\")]"
//@ has  "$.index[?(@.name=='Pair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='Pair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub struct Pair<'a, 'b, T>(&'a T, &'b T);

//@ is   "$.index[?(@.name=='require_pair')].inner.function.generics.params[2].name" '"T"'
//@ has  "$.index[?(@.name=='require_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
//@ has  "$.index[?(@.name=='require_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'b\")]"
//@ has  "$.index[?(@.name=='require_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='require_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub fn require_pair<'a, 'b, T>(_: &'a &'b T) -> Pair<'a, 'b, T> {
    todo!()
}

//@ is   "$.index[?(@.name=='OutlivePair')].inner.struct.generics.params[2].name" '"T"'
//@ has  "$.index[?(@.name=='OutlivePair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
//@ has  "$.index[?(@.name=='OutlivePair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'b\")]"
//@ has  "$.index[?(@.name=='OutlivePair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='OutlivePair')].inner.struct.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub struct OutlivePair<'a, 'b: 'a, T>(&'a T, &'b T);

//@ is   "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[1].name" \"\'b\"
//@ is   "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[2].name" '"T"'
// TODO: enable this: //@ is   "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[1].kind.lifetime.implicitly_outlives" '["\'a"]'
//@ has  "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
//@ has  "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.outlives==\"'b\")]"
//@ has  "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='require_outlive_pair')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub fn require_outlive_pair<'a, 'b, T>(x: &'a T, y: &'b T) -> OutlivePair<'a, 'b, T> {
    todo!()
}
