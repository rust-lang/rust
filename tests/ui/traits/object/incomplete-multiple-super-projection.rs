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
//~^ ERROR conflicting implementations of trait `Trait` for type `(dyn Dyn<(), ()> + 'static)`
    type Assoc = usize;
}

fn call<A, B>(x: usize) -> <dyn Dyn<A, B> as Trait>::Assoc {
    x
}

fn main() {
    let x: &'static str = call::<(), ()>(0xDEADBEEF);
    println!("{x}");
}
