#![feature(non_lifetime_binders, generic_const_exprs)]

fn fun()
where
    for<T = (), const N: usize = 1> ():,
    //~^ ERROR late-bound const parameters cannot be used currently
    //~| ERROR defaults for generic parameters are not allowed in `for<...>` binders
    //~| ERROR defaults for generic parameters are not allowed in `for<...>` binders
{}

fn main() {}
