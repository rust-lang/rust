// Regression test for #137329
#![allow(todo_macro_calls)]

trait B {
    type C<'a>;
    fn d<E>() -> F<E> {
              //~^ ERROR: the trait bound `E: B` is not satisfied
        todo!()
    }
}
struct F<G> {
    h: Option<<G as B>::C>, //~ ERROR: the trait bound `G: B` is not satisfied
    //~^ ERROR missing generics for associated type `B::C`
}

fn main() {}
