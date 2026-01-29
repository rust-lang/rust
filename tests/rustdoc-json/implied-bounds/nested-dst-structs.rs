//! Nested DST-capable structs can still imply `T: Sized` when used by value,
//! even though each layer allows `T: ?Sized`.

pub struct InnerTuple<T: ?Sized>(pub T);
pub struct InnerNamed<T: ?Sized> {
    pub value: T,
}

pub struct OuterTuple<T: ?Sized>(pub T);
pub struct OuterNamed<T: ?Sized> {
    pub value: T,
}

// By-value outer tuple struct requires `InnerTuple<T>` to be Sized, but we don't surface that.
//@ has "$.index[?(@.name=='takes_outer_tuple_value')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='takes_outer_tuple_value')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn takes_outer_tuple_value<T: ?Sized>(value: OuterTuple<InnerTuple<T>>) {
    let _ = value;
}

// By-value outer named struct requires `InnerNamed<T>` to be Sized, but we don't surface that.
//@ has "$.index[?(@.name=='takes_outer_named_value')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='takes_outer_named_value')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn takes_outer_named_value<T: ?Sized>(value: OuterNamed<InnerNamed<T>>) {
    let _ = value;
}

// Indirections allow the outer struct to be a DST, so `T: Sized` is not implied here.
//@ has "$.index[?(@.name=='takes_outer_tuple_ref')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='takes_outer_tuple_ref')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn takes_outer_tuple_ref<T: ?Sized>(value: &OuterTuple<InnerTuple<T>>) {
    let _ = value;
}

// Return-position references also allow DSTs, so `T: Sized` is not implied here either.
//@ has "$.index[?(@.name=='returns_outer_tuple_ref')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='returns_outer_tuple_ref')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn returns_outer_tuple_ref<T: ?Sized>() -> &'static OuterTuple<InnerTuple<T>> {
    todo!()
}

// Raw pointers also allow DSTs, so `T: Sized` is not implied here either.
//@ has "$.index[?(@.name=='takes_outer_named_ptr')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='takes_outer_named_ptr')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn takes_outer_named_ptr<T: ?Sized>(value: *const OuterNamed<InnerNamed<T>>) {
    let _ = value;
}

// Return-position raw pointers also allow DSTs.
//@ has "$.index[?(@.name=='returns_outer_named_ptr')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='returns_outer_named_ptr')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn returns_outer_named_ptr<T: ?Sized>() -> *const OuterNamed<InnerNamed<T>> {
    core::ptr::null()
}
