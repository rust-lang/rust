// Make sure that we check that impl trait types implement the traits that they
// claim to.

#![feature(type_alias_impl_trait)]

pub type X<T> = impl Clone;

#[defines(X)]
fn f<T: Clone>(t: T) -> X<T> {
    t
    //~^ ERROR the trait bound `T: Clone` is not satisfied
}

fn g<T>(o: Option<X<T>>) -> Option<X<T>> {
    o.clone()
}

fn main() {
    g(None::<X<&mut ()>>);
}
