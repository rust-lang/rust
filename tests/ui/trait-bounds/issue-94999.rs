//@ check-pass

trait Identity<Q> {
    type T;
}

impl<Q, T> Identity<Q> for T {
    type T = T;
}

trait Holds {
    type Q;
}

struct S;
struct X(S);

struct XHelper;

impl Holds for X {
    type Q = XHelper;
}

impl<Q> Clone for X
where
    <S as Identity<Q>>::T: Clone,
    X: Holds<Q = Q>,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

fn main() {}
