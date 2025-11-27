pub trait SizedOnly: Sized {}
impl<T: Sized> SizedOnly for T {}

pub trait StaticOnly: 'static {}
impl<T: 'static> StaticOnly for T {}

pub trait Container {
    //@ has "$.index[?(@.name=='Item')].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
    //@ has "$.index[?(@.name=='Item')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    //@ !has "$.index[?(@.name=='Item')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
    type Item: SizedOnly + ?Sized;

    //@ has "$.index[?(@.name=='UnsizedItem')].inner.assoc_type.bounds[?(@.trait_bound.modifier=='maybe' && @.trait_bound.trait.path=='Sized')]"
    //@ !has "$.index[?(@.name=='UnsizedItem')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
    type UnsizedItem: ?Sized;
}

//@ has "$.index[?(@.name=='Output')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ has "$.index[?(@.name=='Output')].inner.assoc_type.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='Output')].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
pub trait StaticContainer {
    type Output: StaticOnly;
}
