// Make sure that if there are multiple applicable bounds on a projection, we
// consider them ambiguous. In this test we are initially trying to solve
// `Self::Repr: From<_>`, which is ambiguous until we later infer `_` to
// `{integer}`.

//@ check-pass

trait PrimeField: Sized {
    type Repr: From<u64> + From<Self>;
    type Repr2: From<Self> + From<u64>;

    fn method() {
        Self::Repr::from(10);
        Self::Repr2::from(10);
    }
}

fn function<T: PrimeField>() {
    T::Repr::from(10);
    T::Repr2::from(10);
}

fn main() {}
