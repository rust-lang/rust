#![feature(non_lifetime_binders, generic_const_exprs)]
//~^ WARN the feature `non_lifetime_binders` is incomplete
//~| WARN the feature `generic_const_exprs` is incomplete

fn fun()
where
    for<T = (), const N: usize = 1> ():,
    //~^ ERROR late-bound const parameters cannot be used currently
    //~| ERROR defaults for generic parameters are not allowed in `for<...>` binders
    //~| ERROR defaults for generic parameters are not allowed in `for<...>` binders
{}

fn main() {}
