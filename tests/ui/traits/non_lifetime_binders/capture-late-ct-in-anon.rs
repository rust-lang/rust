#![feature(non_lifetime_binders, generic_const_exprs)]
//~^ WARN the feature `non_lifetime_binders` is incomplete
//~| WARN the feature `generic_const_exprs` is incomplete

fn a()
where
    for<const C: usize> [(); C + 1]: Copy,
    //~^ ERROR cannot capture late-bound const parameter in a constant
{    
}

fn b()
where
    for<const C: usize> [(); C]: Copy,
{
}

fn main() {}
