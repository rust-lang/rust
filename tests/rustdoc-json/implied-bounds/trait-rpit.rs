pub trait StaticOnly: 'static {}
impl<T: 'static> StaticOnly for T {}

pub trait SizedOnly: Sized {}
impl<T: Sized> SizedOnly for T {}

pub trait Provider {
    //@ has "$.index[?(@.name=='provides_static')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
    //@ has "$.index[?(@.name=='provides_static')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    //@ has "$.index[?(@.name=='provides_static')].inner.function.sig.output.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
    //@ !has "$.index[?(@.name=='provides_static')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
    fn provides_static(&self) -> impl StaticOnly;

    //@ has "$.index[?(@.name=='provides_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
    //@ has "$.index[?(@.name=='provides_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
    //@ has "$.index[?(@.name=='provides_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    //@ !has "$.index[?(@.name=='provides_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
    fn provides_ref(&self) -> &(impl SizedOnly + ?Sized);
}
