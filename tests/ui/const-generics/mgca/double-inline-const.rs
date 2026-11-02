#![feature(gca_min_const_items)]

struct S<const N: usize>;

impl<const N: usize> S<N> {
    const Q: usize = 2;
    fn foo(_: S<{ const { const { Self::Q } } }>) {}
    //~^ ERROR generic `Self` types are currently not permitted in anonymous constants
}

fn main() {}
