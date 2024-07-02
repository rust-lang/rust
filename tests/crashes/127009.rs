//@ known-bug: #127009

#![feature(non_lifetime_binders)]

fn b()
where
    for<const C: usize> [(); C]: Copy,
{
}

fn main() {}
