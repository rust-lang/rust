#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn b()
where
    for<const C: usize> [(); C]: Copy,
    //~^ ERROR late-bound const parameters cannot be used currently
{
}

fn main() {}
