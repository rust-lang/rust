// Make sure that we check that impl trait types implement the traits that they
// claim to.

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

type X<T> = impl Clone;
//~^ ERROR the trait bound `T: Clone` is not satisfied

fn f<T: Clone>(t: T) -> X<T> {
    t
}

fn g<T>(o: Option<X<T>>) -> Option<X<T>> {
    o.clone()
}

fn main() {
    g(None::<X<&mut ()>>);
}
