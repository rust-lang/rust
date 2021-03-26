// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

trait Trait {}
type Underconstrained<T: Trait> = impl 'static;
//~^ ERROR: at least one trait must be specified

// no `Trait` bound
fn underconstrain<T>(_: T) -> Underconstrained<T> {
    //~^ ERROR the trait bound `T: Trait`
    unimplemented!()
}
