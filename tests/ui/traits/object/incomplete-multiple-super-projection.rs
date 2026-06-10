// Regression test for #133361.

trait Sup<T> {
    type Assoc;
}

impl<T> Sup<T> for () {
    type Assoc = T;
}
impl<T, U> Dyn<T, U> for () {}

trait Dyn<A, B>: Sup<A, Assoc = A> + Sup<B, Assoc = B> {}

trait Trait {
    type Assoc;
}
impl Trait for dyn Dyn<(), ()> {
    type Assoc = &'static str;
}
impl<A, B> Trait for dyn Dyn<A, B> {
    //~^ ERROR conflicting associated type bindings for `Assoc`
    type Assoc = usize;
}

fn call<A, B>(x: usize) -> <dyn Dyn<A, B> as Trait>::Assoc {
    //~^ ERROR conflicting associated type bindings for `Assoc`
    //~| ERROR conflicting associated type bindings for `Assoc`
    //~| ERROR conflicting associated type bindings for `Assoc`
    //~| ERROR the trait bound `(dyn Dyn<A, B> + 'static): Trait` is not satisfied
    //~| ERROR the trait bound `(dyn Dyn<A, B> + 'static): Trait` is not satisfied
    x
}

fn main() {
    let x: &'static str = call::<(), ()>(0xDEADBEEF);
    println!("{x}");
}
