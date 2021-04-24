// Regression test for issue #68368
// Ensures that we don't ICE when emitting an error
// for a non-defining use when lifetimes are involved

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
trait Trait<T> {}
type Alias<'a, U> = impl Trait<U>;
fn f<'a>() -> Alias<'a, ()> {}
//~^ ERROR non-defining opaque type use in defining scope

fn main() {}

impl Trait<()> for () {}
