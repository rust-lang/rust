//@ check-pass

// We previously incorrectly deduplicated the list of projection bounds
// of trait objects, causing us to incorrectly reject this code, cc #136458.

trait Sup<T> {
    type Assoc;
}

impl<T> Sup<T> for () {
    type Assoc = T;
}

trait Trait<A, B>: Sup<A, Assoc = A> + Sup<B, Assoc = B> {}

impl<T, U> Trait<T, U> for () {}

fn main() {
    let x: &dyn Trait<(), _> = &();
    let y: &dyn Trait<_, ()> = x;
}
