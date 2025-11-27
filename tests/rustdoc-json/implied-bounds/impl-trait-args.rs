use std::fmt::Debug;

pub trait SizedOnly: Sized {}
impl<T: Sized> SizedOnly for T {}

pub trait StaticOnly: 'static {}
impl<T: 'static> StaticOnly for T {}

//@ is "$.index[?(@.name=='claimed_unsized_value_arg')].inner.function.sig.inputs[0][0]" '"arg"'
//@ has "$.index[?(@.name=='claimed_unsized_value_arg')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='claimed_unsized_value_arg')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ has "$.index[?(@.name=='claimed_unsized_value_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='claimed_unsized_value_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
pub fn claimed_unsized_value_arg(arg: impl SizedOnly + ?Sized) {}

//@ is "$.index[?(@.name=='claimed_unsized_ref_arg')].inner.function.sig.inputs[0][0]" '"arg"'
//@ has "$.index[?(@.name=='claimed_unsized_ref_arg')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='claimed_unsized_ref_arg')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='claimed_unsized_ref_arg')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
pub fn claimed_unsized_ref_arg(arg: &(impl SizedOnly + ?Sized)) {}

//@ is "$.index[?(@.name=='implicitly_static_value_arg')].inner.function.sig.inputs[0][0]" '"arg"'
//@ has "$.index[?(@.name=='implicitly_static_value_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='implicitly_static_value_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
pub fn implicitly_static_value_arg(arg: impl StaticOnly) {}

// By-value `impl Trait + ?Sized` does not add an implied `Sized` bound unless it's from the trait.
//@ is "$.index[?(@.name=='implicitly_sized_because_fn_arg')].inner.function.sig.inputs[0][0]" '"arg"'
//@ !has "$.index[?(@.name=='implicitly_sized_because_fn_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='implicitly_sized_because_fn_arg')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Debug')]"
pub fn implicitly_sized_because_fn_arg(arg: impl Debug + ?Sized) {}

//@ has "$.index[?(@.name=='sized_only_ref')].inner.function.sig.inputs[0][0]" '"arg"'
//@ has "$.index[?(@.name=='sized_only_ref')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
//@ has "$.index[?(@.name=='sized_only_ref')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='sized_only_ref')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub fn sized_only_ref(arg: &(impl SizedOnly + ?Sized)) {
    let _ = arg;
}
