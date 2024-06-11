pub trait Parser<E> {
    fn parse(&self) -> E;
}

impl<E, T: Fn() -> E> Parser<E> for T {
    fn parse(&self) -> E {
        self()
    }
}

pub fn recursive_fn<E>() -> impl Parser<E> {
    //~^ ERROR: cycle detected
    move || recursive_fn().parse()
    //~^ ERROR: type annotations needed
    //~| ERROR: no method named `parse` found for opaque type
}

fn main() {}
