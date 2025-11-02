pub trait SizedOnly: Sized {}
impl<T: Sized> SizedOnly for T {}

pub trait StaticOnly: 'static {}
impl<T: 'static> StaticOnly for T {}

//@ has "$.index[?(@.name=='Item')].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ has "$.index[?(@.name=='Item')].inner.assoc_type.bounds[?(@.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='Item')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='Item')].inner.assoc_type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='Item')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
pub trait Container {
    type Item: SizedOnly + ?Sized;
}

//@ has "$.index[?(@.name=='Output')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='Output')].inner.assoc_type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='Output')].inner.assoc_type.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='Output')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
pub trait StaticContainer {
    type Output: StaticOnly;
}
