#![feature(type_alias_impl_trait)]

fn main() {}

trait Trait {}
type Underconstrained<T: Trait> = impl Send;

// no `Trait` bound
fn underconstrain<T>(_: T) -> Underconstrained<T> {
    //~^ ERROR trait `Trait` is not implemented for `T`
    //~| ERROR trait `Trait` is not implemented for `T`
    unimplemented!()
}
