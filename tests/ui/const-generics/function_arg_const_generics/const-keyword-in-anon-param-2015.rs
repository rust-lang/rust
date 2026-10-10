//@ edition: 2015
#![feature(function_arg_const_generics, gca_min_const_items, const_trait_impl)]

#[cfg(false)]
trait DynType {
    fn f(const dyn Trait); //~ ERROR
}

#[cfg(false)]
trait ImplType {
    fn f(const impl Trait); //~ ERROR
}

#[cfg(false)]
trait FnPtrType {
    fn f(const fn()); //~ ERROR
}

trait ConstTraitBound {
    fn f(const Fn()); //~ ERROR const trait bounds are not allowed in trait object types
    //~| WARN anonymous parameters are deprecated
    //~| WARN this is accepted in the current edition
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
}

fn main() {}
