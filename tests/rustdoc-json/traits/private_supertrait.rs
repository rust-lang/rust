//@ !has "$.index[?(@.name == 'sealed')]"
mod sealed {
    //@ set sealed_id = "$.index[?(@.name=='Sealed')].id"
    pub trait Sealed {}
}

//@ count "$.index[?(@.name=='Trait')].inner.trait.bounds[*]" 1
//@ is    "$.index[?(@.name=='Trait')].inner.trait.bounds[0].trait_bound.trait.id" $sealed_id
pub trait Trait: sealed::Sealed {}
