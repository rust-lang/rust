//@ check-pass

struct S;

enum E {
    V,
}

type A = E;

fn main() {
    let mut a;

    S = S;
    (S, a) = (S, ());

    E::V = E::V;
    (E::V, a) = (E::V, ());

    <E>::V = E::V;
    (<E>::V, a) = (E::V, ());
    A::V = A::V;
    (A::V, a) = (E::V, ());
}

impl S {
    fn check() {
        let a;
        Self = S;
        (Self, a) = (S, ());
    }
}

impl E {
    fn check() {
        let a;
        Self::V = E::V;
        (Self::V, a) = (E::V, ());
    }
}
