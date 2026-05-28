#![feature(non_lifetime_binders)]

fn b()
where
    for<const C: usize> [(); C]: Copy,
    //~^ ERROR late-bound const parameters cannot be used currently
{
}

fn main() {}
