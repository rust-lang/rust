#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn b()
where
    for<const C: usize> [(); C]: Copy,
    //~^ ERROR cannot capture late-bound const parameter in a constant
{
}

fn main() {}
