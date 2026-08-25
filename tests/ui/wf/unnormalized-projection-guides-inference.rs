// The WF requirements of the *unnormalized* form of type annotations
// can guide inference.
//@ check-pass

pub trait EqualTo {
    type Ty;
}
impl<X> EqualTo for X {
    type Ty = X;
}

trait MyTrait<U: EqualTo<Ty = Self>> {
    type Out;
}
impl<T, U: EqualTo<Ty = T>> MyTrait<U> for T {
    type Out = ();
}

fn main() {
    let _: <_ as MyTrait<u8>>::Out;
    // We shoud be able to infer a value for the inference variable above.
    // The WF of the unnormalized projection requires `u8: EqualTo<Ty = _>`,
    // which is sufficient to guide inference.
}
