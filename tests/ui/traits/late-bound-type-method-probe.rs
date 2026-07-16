// A method probe involving a late-bound type parameter should report normal
// diagnostics instead of ICEing while structurally normalizing obligations.

#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

pub trait T {
    fn t<Tail>(&self, _: F) {}
    //~^ ERROR cannot find type `F` in this scope
}

pub fn crash<V>(v: &V)
where
    for<F> F: T + 'static,
{
    v.t(|| {});
}

fn main() {}
