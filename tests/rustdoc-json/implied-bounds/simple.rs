//@ has "$.index[?(@.name=='unsized_ok')]"
//@ has "$.index[?(@.name=='sized_through_trait')]"
//@ has "$.index[?(@.name=='static_via_trait')]"
//@ has "$.index[?(@.name=='carries_inner_unsized')]"

pub trait NeedsSized: Sized {}

//@ count "$.index[?(@.name=='unsized_ok')].inner.function.generics.params[0].kind.type.implied_bounds[*]" 0
pub fn unsized_ok<T: ?Sized>(_value: &T) {}

//@ has   "$.index[?(@.name=='sized_through_trait')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has   "$.index[?(@.name=='sized_through_trait')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has   "$.index[?(@.name=='sized_through_trait')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has  "$.index[?(@.name=='sized_through_trait')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
pub fn sized_through_trait<T: NeedsSized + ?Sized>(_value: &T) {}

pub trait NeedsStatic: 'static {}
impl<T: 'static> NeedsStatic for T {}

//@ has   "$.index[?(@.name=='static_via_trait')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ has   "$.index[?(@.name=='static_via_trait')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has  "$.index[?(@.name=='static_via_trait')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='NeedsStatic')]"
pub fn static_via_trait<T: NeedsStatic>(_value: T) {}

//@ count "$.index[?(@.name=='explicit_sized')].inner.function.generics.params[0].kind.type.implied_bounds[*]" 0
pub fn explicit_sized<T: Sized>(_value: T) {}

//@ count "$.index[?(@.name=='explicit_sized_ref')].inner.function.generics.params[0].kind.type.implied_bounds[*]" 0
pub fn explicit_sized_ref<T: Sized>(_value: &T) {}

//@ has   "$.index[?(@.name=='Inner')].inner.struct.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ has   "$.index[?(@.name=='Inner')].inner.struct.generics.params[1].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
pub struct Inner<'a, T>(&'a T);

//@ has   "$.index[?(@.name=='InnerUnsized')].inner.struct.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has  "$.index[?(@.name=='InnerUnsized')].inner.struct.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ has   "$.index[?(@.name=='InnerUnsized')].inner.struct.generics.params[1].kind.type.implied_bounds[?(@.outlives==\"'a\")]"
pub struct InnerUnsized<'a, T: ?Sized>(&'a T);

//@ has   "$.index[?(@.name=='carries_inner_unsized')].inner.function.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub fn carries_inner_unsized<'a, T>(lt: &'a i64, ty: T) -> InnerUnsized<'a, T> {
    let _ = lt;
    let _ = ty;
    todo!()
}
