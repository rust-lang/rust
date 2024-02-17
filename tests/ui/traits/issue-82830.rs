//@ check-pass

trait A<Y, N> {
    type B;
}

type MaybeBox<T> = <T as A<T, Box<T>>>::B;
struct P {
    t: MaybeBox<P>,
}

impl<Y, N> A<Y, N> for P {
    type B = N;
}

fn main() {
    let t: MaybeBox<P>;
}
