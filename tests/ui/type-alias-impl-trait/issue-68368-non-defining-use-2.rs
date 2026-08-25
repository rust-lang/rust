// Regression test for issue #68368
// Ensures that we don't ICE when emitting an error
// for a non-defining use when lifetimes are involved

#![feature(type_alias_impl_trait)]
trait Trait<T> {}
type Alias<'a, U> = impl Trait<U>;

#[define_opaque(Alias)]
fn f<'a>() -> Alias<'a, ()> {}
//~^ ERROR expected generic type parameter, found `()`

fn main() {}

impl<X> Trait<X> for () {}
