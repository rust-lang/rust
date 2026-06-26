#![feature(type_alias_impl_trait)]

fn main() {}

trait Trait {}
type Underconstrained<T: Trait> = impl Send;

// no `Trait` bound
#[define_opaque(Underconstrained)]
fn underconstrain<T>(_: T) -> Underconstrained<T> {
    //~^ ERROR the trait bound `T: Trait`
    //~| ERROR the trait bound `T: Trait`
    unimplemented!()
}
