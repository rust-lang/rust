#![no_std]

//@ has "$.index[?(@.name=='FooDynIncompatible')]"
//@ is "$.index[?(@.name=='FooDynIncompatible')].inner.trait.is_dyn_compatible" false
pub trait FooDynIncompatible {
    fn foo() -> Self;
}

//@ has "$.index[?(@.name=='BarDynIncompatible')]"
//@ is "$.index[?(@.name=='BarDynIncompatible')].inner.trait.is_dyn_compatible" false
pub trait BarDynIncompatible<T> {
    fn foo(i: T);
}

//@ has "$.index[?(@.name=='FooDynCompatible')]"
//@ is "$.index[?(@.name=='FooDynCompatible')].inner.trait.is_dyn_compatible" true
pub trait FooDynCompatible {
    fn foo(&self);
}
