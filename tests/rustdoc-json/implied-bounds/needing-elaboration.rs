pub trait NeedsSized: Sized {}
pub trait IndirectSized: NeedsSized {}

pub trait NeedsStatic: 'static {}
pub trait IndirectStatic: NeedsStatic {}

// - `IndirectStatic` and `IndirectSized` are explicit bounds, only appearing in `bounds`.
// - `NeedsSized`, `NeedsStatic`, `Sized`, and `'static` are implied bounds,
//   only appearing in `implied_bounds`.
//@ has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='IndirectStatic')]"
//@ has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='IndirectSized')]"
//@ !has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='NeedsStatic')]"
//@ !has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ !has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='IndirectStatic')]"
//@ !has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='IndirectSized')]"
//@ has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='NeedsStatic')]"
//@ has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ has "$.index[?(@.name=='example')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.outlives==\"'static\")]"
pub fn example<T: IndirectStatic + IndirectSized>(value: &T) {}
