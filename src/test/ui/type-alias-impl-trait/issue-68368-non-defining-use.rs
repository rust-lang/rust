// Regression test for issue #68368
// Ensures that we don't ICE when emitting an error
// for a non-defining use when lifetimes are involved

#![feature(type_alias_impl_trait)]
trait Trait<T> {}
type Alias<'a, U> = impl Trait<U>;
//~^ ERROR could not find defining uses
fn f<'a>() -> Alias<'a, ()> {}
//~^ ERROR non-defining opaque type use in defining scope

fn main() {}

impl Trait<()> for () {}
