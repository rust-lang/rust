#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct S<A, B>(A, B);
struct Featureless;

trait One {}
trait Two {}

impl<T: One> S<Featureless, T> {
    type X = ();
}

impl<T: Two> S<T, Featureless> {
    type X = String;
}

fn main() {
    let _: S::<Featureless, Featureless>::X; //~ ERROR the associated type `X` exists for `S<Featureless, Featureless>`, but its trait bounds were not satisfied
}
