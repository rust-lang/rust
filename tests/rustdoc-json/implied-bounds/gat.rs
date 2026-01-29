pub trait SizedOnly: Sized {}

impl<T: Sized> SizedOnly for T {}

pub trait StaticOnly: 'static {}

impl<T: 'static> StaticOnly for T {}

pub trait GatContainer {
    //@ count "$.index[?(@.name=='Plain' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[*]" 2
    //@ has "$.index[?(@.name=='Plain' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    type Plain<'a, T>;

    //@ has "$.index[?(@.name=='MaybeUnsized' && @.inner.assoc_type.type==null)].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
    //@ count "$.index[?(@.name=='MaybeUnsized' && @.inner.assoc_type.type==null)].inner.assoc_type.implied_bounds[*]" 0
    //@ has "$.index[?(@.name=='MaybeUnsized' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
    //@ count "$.index[?(@.name=='MaybeUnsized' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[*]" 0
    type MaybeUnsized<'a, T: ?Sized>: ?Sized;

    //@ has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type==null)].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
    //@ has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type==null)].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
    //@ has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type==null)].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    //@ has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
    //@ has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
    //@ has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    //@ !has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
    type MaybeUnsizedButSized<'a, T: SizedOnly + ?Sized>: SizedOnly + ?Sized;

    //@ has "$.index[?(@.name=='Static' && @.inner.assoc_type.type==null)].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
    //@ has "$.index[?(@.name=='Static' && @.inner.assoc_type.type==null)].inner.assoc_type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    //@ has "$.index[?(@.name=='Static' && @.inner.assoc_type.type==null)].inner.assoc_type.implied_bounds[?(@.outlives==\"'static\")]"
    //@ has "$.index[?(@.name=='Static' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
    //@ has "$.index[?(@.name=='Static' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    //@ has "$.index[?(@.name=='Static' && @.inner.assoc_type.type==null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[?(@.outlives==\"'static\")]"
    type Static<'a, T: StaticOnly>: StaticOnly;
}

pub struct UsesArray;

impl GatContainer for UsesArray {
    //@ has "$.index[?(@.name=='Plain' && @.inner.assoc_type.type!=null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    type Plain<'a, T> = (&'a (), [T; 1]);

    //@ count "$.index[?(@.name=='MaybeUnsized' && @.inner.assoc_type.type!=null)].inner.assoc_type.implied_bounds[*]" 0
    //@ has "$.index[?(@.name=='MaybeUnsized' && @.inner.assoc_type.type!=null)].inner.assoc_type.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
    //@ count "$.index[?(@.name=='MaybeUnsized' && @.inner.assoc_type.type!=null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[*]" 0
    type MaybeUnsized<'a, T: ?Sized> = *const T;

    //@ has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type!=null)].inner.assoc_type.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='SizedOnly')]"
    //@ has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type!=null)].inner.assoc_type.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
    //@ has "$.index[?(@.name=='MaybeUnsizedButSized' && @.inner.assoc_type.type!=null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    type MaybeUnsizedButSized<'a, T: SizedOnly + ?Sized> = [T; 1];

    //@ has "$.index[?(@.name=='Static' && @.inner.assoc_type.type!=null)].inner.assoc_type.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
    //@ has "$.index[?(@.name=='Static' && @.inner.assoc_type.type!=null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
    //@ has "$.index[?(@.name=='Static' && @.inner.assoc_type.type!=null)].inner.assoc_type.generics.params[1].kind.type.implied_bounds[?(@.outlives==\"'static\")]"
    type Static<'a, T: StaticOnly> = T;
}
