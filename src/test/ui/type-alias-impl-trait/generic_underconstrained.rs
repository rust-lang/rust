#![feature(type_alias_impl_trait)]

fn main() {}

trait Trait {}
type Underconstrained<T: Trait> = impl 'static; //~ ERROR the trait bound `T: Trait`
//~^ ERROR: at least one trait must be specified

// no `Trait` bound
fn underconstrain<T>(_: T) -> Underconstrained<T> {
    unimplemented!()
}
