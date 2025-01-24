trait Sup<T> {
    type Assoc: Default;
}

impl<T: Default> Sup<T> for () {
    type Assoc = T;
}
impl<T: Default, U: Default> Dyn<T, U> for () {}

trait Dyn<A, B>: Sup<A, Assoc = A> + Sup<B> {}

fn main() {
    let q: <dyn Dyn<i32, u32> as Sup<u32>>::Assoc = Default::default();
    //~^ ERROR the value of the associated type `Assoc` in `Sup<u32>` must be specified
}
