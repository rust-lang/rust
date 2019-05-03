#![feature(existential_type)]

fn main() {}

trait Trait {}
existential type Underconstrained<T: Trait>: 'static; //~ ERROR the trait bound `T: Trait`
//~^ ERROR: at least one trait must be specified

// no `Trait` bound
fn underconstrain<T>(_: T) -> Underconstrained<T> {
    unimplemented!()
}
