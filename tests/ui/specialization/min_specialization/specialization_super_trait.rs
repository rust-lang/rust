// Test that supertraits can't be assumed in impls of
// `rustc_specialization_trait`, as such impls would
// allow specializing on the supertrait.

#![feature(min_specialization)]
#![feature(rustc_attrs)]

#[rustc_specialization_trait]
trait SpecMarker: Default {
    fn f();
}

impl<T: Default> SpecMarker for T {
    //~^ ERROR cannot specialize
    fn f() {}
}

fn main() {}
