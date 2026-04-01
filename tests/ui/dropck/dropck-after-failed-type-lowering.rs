// Regression test for #137329

trait B {
    type C<'a>;
    fn d<E>() -> F<E> {
        todo!()
    }
}
struct F<G> {
    h: Option<<G as B>::C>,
    //~^ ERROR missing generics for associated type `B::C`
}

fn main() {}
