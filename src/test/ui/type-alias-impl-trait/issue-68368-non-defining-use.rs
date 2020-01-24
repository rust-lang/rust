// Regression test for issue #68368
// Ensures that we don't ICE when emitting an error
// for a non-defining use when lifetimes are involved

#![feature(type_alias_impl_trait)]
trait Trait<T> {}
type Alias<'a, U> = impl Trait<U>; //~ ERROR could not find defining uses
fn f<'a>() -> Alias<'a, ()> {}
//~^ ERROR defining opaque type use does not fully define opaque type: generic parameter `U`

fn main() {}

impl Trait<()> for () {}
