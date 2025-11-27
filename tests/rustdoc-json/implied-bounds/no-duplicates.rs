pub trait SizedOnly: Sized {}
impl<T: Sized> SizedOnly for T {}

pub trait StaticOnly: 'static {}
impl<T: 'static> StaticOnly for T {}

pub trait OtherSized: Sized {}
impl<T: Sized> OtherSized for T {}

pub trait OtherStatic: 'static {}
impl<T: 'static> OtherStatic for T {}

//@ has "$.index[?(@.name=='duplicate_generic_sized')]"
//@ count "$.index[?(@.name=='duplicate_generic_sized')].inner.function.generics.params[0].kind.type.bounds[*]" 2
//@ has  "$.index[?(@.name=='duplicate_generic_sized')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ has  "$.index[?(@.name=='duplicate_generic_sized')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='duplicate_generic_sized')].inner.function.generics.params[0].kind.type.implied_bounds[*]"
pub fn duplicate_generic_sized<T: SizedOnly + Sized>(_t: T) {}

//@ has "$.index[?(@.name=='duplicate_where_bounds_only')]"
//@ count "$.index[?(@.name=='duplicate_where_bounds_only')].inner.function.generics.params[0].kind.type.bounds[*]" 0
//@ count "$.index[?(@.name=='duplicate_where_bounds_only')].inner.function.generics.where_predicates[0].bound_predicate.bounds[?(@.trait_bound.trait.path=='SizedOnly')]" 2
//@ count "$.index[?(@.name=='duplicate_where_bounds_only')].inner.function.generics.params[0].kind.type.implied_bounds[*]" 1
//@ has  "$.index[?(@.name=='duplicate_where_bounds_only')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn duplicate_where_bounds_only<T>(_t: T)
where
    T: SizedOnly + SizedOnly,
{
}

//@ has "$.index[?(@.name=='duplicate_generic_static')]"
//@ count "$.index[?(@.name=='duplicate_generic_static')].inner.function.generics.params[0].kind.type.bounds[*]" 2
//@ has  "$.index[?(@.name=='duplicate_generic_static')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has  "$.index[?(@.name=='duplicate_generic_static')].inner.function.generics.params[0].kind.type.bounds[?(@.outlives==\"'static\")]"
//@ count "$.index[?(@.name=='duplicate_generic_static')].inner.function.generics.params[0].kind.type.implied_bounds[*]" 1
//@ has  "$.index[?(@.name=='duplicate_generic_static')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='duplicate_generic_static')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ !has "$.index[?(@.name=='duplicate_generic_static')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.outlives==\"'static\")]"
pub fn duplicate_generic_static<T: StaticOnly + 'static>(_t: T) {}

//@ count "$.index[?(@.name=='SizedItem')].inner.assoc_type.bounds[*]" 2
//@ has  "$.index[?(@.name=='SizedItem')].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ has  "$.index[?(@.name=='SizedItem')].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='SizedItem')].inner.assoc_type.implied_bounds[*]"
pub trait WithSizedAssoc {
    type SizedItem: SizedOnly + Sized;
}

//@ count "$.index[?(@.name=='StaticItem')].inner.assoc_type.bounds[*]" 2
//@ has  "$.index[?(@.name=='StaticItem')].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has  "$.index[?(@.name=='StaticItem')].inner.assoc_type.bounds[?(@.outlives==\"'static\")]"
//@ count "$.index[?(@.name=='StaticItem')].inner.assoc_type.implied_bounds[*]" 1
//@ has  "$.index[?(@.name=='StaticItem')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='StaticItem')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ !has "$.index[?(@.name=='StaticItem')].inner.assoc_type.implied_bounds[?(@.outlives==\"'static\")]"
pub trait WithStaticAssoc {
    type StaticItem: StaticOnly + 'static;
}

//@ has "$.index[?(@.name=='sized_not_implied')]"
//@ count "$.index[?(@.name=='sized_not_implied')].inner.function.sig.inputs[0][1].impl_trait.bounds[*]" 2
//@ has  "$.index[?(@.name=='sized_not_implied')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ has  "$.index[?(@.name=='sized_not_implied')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ count "$.index[?(@.name=='sized_not_implied')].inner.function.generics.params[0].kind.type.bounds[*]" 2
//@ has  "$.index[?(@.name=='sized_not_implied')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ has  "$.index[?(@.name=='sized_not_implied')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='sized_not_implied')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[*]"
//@ !has "$.index[?(@.name=='sized_not_implied')].inner.function.generics.params[0].kind.type.implied_bounds[*]"
pub fn sized_not_implied(arg: impl SizedOnly + Sized) {
    let _ = arg;
}

//@ has "$.index[?(@.name=='static_not_implied')]"
//@ count "$.index[?(@.name=='static_not_implied')].inner.function.sig.inputs[0][1].impl_trait.bounds[*]" 2
//@ has  "$.index[?(@.name=='static_not_implied')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has  "$.index[?(@.name=='static_not_implied')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.outlives==\"'static\")]"
//@ count "$.index[?(@.name=='static_not_implied')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[*]" 1
//@ has  "$.index[?(@.name=='static_not_implied')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ count "$.index[?(@.name=='static_not_implied')].inner.function.generics.params[0].kind.type.bounds[*]" 2
//@ has  "$.index[?(@.name=='static_not_implied')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has  "$.index[?(@.name=='static_not_implied')].inner.function.generics.params[0].kind.type.bounds[?(@.outlives==\"'static\")]"
//@ count "$.index[?(@.name=='static_not_implied')].inner.function.generics.params[0].kind.type.implied_bounds[*]" 1
//@ has  "$.index[?(@.name=='static_not_implied')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='static_not_implied')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ !has "$.index[?(@.name=='static_not_implied')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
pub fn static_not_implied(arg: impl StaticOnly + 'static) {
    let _ = arg;
}

//@ has "$.index[?(@.name=='sized_return_not_implied')]"
//@ count "$.index[?(@.name=='sized_return_not_implied')].inner.function.sig.output.impl_trait.bounds[*]" 2
//@ has  "$.index[?(@.name=='sized_return_not_implied')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ has  "$.index[?(@.name=='sized_return_not_implied')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='sized_return_not_implied')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ !has "$.index[?(@.name=='sized_return_not_implied')].inner.function.sig.output.impl_trait.implied_bounds[*]"
pub fn sized_return_not_implied() -> impl SizedOnly + Sized {
    ()
}

//@ has "$.index[?(@.name=='static_return_not_implied')]"
//@ count "$.index[?(@.name=='static_return_not_implied')].inner.function.sig.output.impl_trait.bounds[*]" 2
//@ has  "$.index[?(@.name=='static_return_not_implied')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has  "$.index[?(@.name=='static_return_not_implied')].inner.function.sig.output.impl_trait.bounds[?(@.outlives==\"'static\")]"
//@ count "$.index[?(@.name=='static_return_not_implied')].inner.function.sig.output.impl_trait.implied_bounds[*]" 1
//@ has  "$.index[?(@.name=='static_return_not_implied')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='static_return_not_implied')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ !has "$.index[?(@.name=='static_return_not_implied')].inner.function.sig.output.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
pub fn static_return_not_implied() -> impl StaticOnly + 'static {
    ()
}

//@ has "$.index[?(@.name=='diamond_implied_sized_arg')]"
//@ count "$.index[?(@.name=='diamond_implied_sized_arg')].inner.function.sig.inputs[0][1].impl_trait.bounds[*]" 2
//@ has  "$.index[?(@.name=='diamond_implied_sized_arg')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ has  "$.index[?(@.name=='diamond_implied_sized_arg')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='OtherSized')]"
//@ count "$.index[?(@.name=='diamond_implied_sized_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[*]" 1
//@ has  "$.index[?(@.name=='diamond_implied_sized_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn diamond_implied_sized_arg(arg: impl SizedOnly + OtherSized) {
    let _ = arg;
}

//@ has "$.index[?(@.name=='diamond_implied_static_arg')]"
//@ count "$.index[?(@.name=='diamond_implied_static_arg')].inner.function.sig.inputs[0][1].impl_trait.bounds[*]" 2
//@ has  "$.index[?(@.name=='diamond_implied_static_arg')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has  "$.index[?(@.name=='diamond_implied_static_arg')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='OtherStatic')]"
//@ count "$.index[?(@.name=='diamond_implied_static_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[*]" 2
//@ has  "$.index[?(@.name=='diamond_implied_static_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ has  "$.index[?(@.name=='diamond_implied_static_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
pub fn diamond_implied_static_arg(arg: impl StaticOnly + OtherStatic) {
    let _ = arg;
}

//@ has "$.index[?(@.name=='diamond_implied_sized_return')]"
//@ count "$.index[?(@.name=='diamond_implied_sized_return')].inner.function.sig.output.impl_trait.bounds[*]" 2
//@ has  "$.index[?(@.name=='diamond_implied_sized_return')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ has  "$.index[?(@.name=='diamond_implied_sized_return')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='OtherSized')]"
//@ count "$.index[?(@.name=='diamond_implied_sized_return')].inner.function.sig.output.impl_trait.implied_bounds[*]" 1
//@ has  "$.index[?(@.name=='diamond_implied_sized_return')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn diamond_implied_sized_return() -> impl SizedOnly + OtherSized {
    ()
}

//@ has "$.index[?(@.name=='diamond_implied_static_return')]"
//@ count "$.index[?(@.name=='diamond_implied_static_return')].inner.function.sig.output.impl_trait.bounds[*]" 2
//@ has  "$.index[?(@.name=='diamond_implied_static_return')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has  "$.index[?(@.name=='diamond_implied_static_return')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='OtherStatic')]"
//@ count "$.index[?(@.name=='diamond_implied_static_return')].inner.function.sig.output.impl_trait.implied_bounds[*]" 2
//@ has  "$.index[?(@.name=='diamond_implied_static_return')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ has  "$.index[?(@.name=='diamond_implied_static_return')].inner.function.sig.output.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
pub fn diamond_implied_static_return() -> impl StaticOnly + OtherStatic {
    ()
}
