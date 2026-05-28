struct Invariant<'a>(&'a mut &'a mut ());

trait Trait {
    fn hello(self_: Invariant<'_>) -> impl Sized + use<Self>;
}

impl Trait for () {
    fn hello(self_: Invariant<'_>) -> impl Sized + use<'_> {}
    //~^ ERROR return type captures more lifetimes than trait definition
}

fn main() {}
