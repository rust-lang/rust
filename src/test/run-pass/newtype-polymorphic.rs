tag myvec<X> = [X];

fn myvec_deref<X: copy>(mv: myvec<X>) -> [X] { ret *mv; }

fn myvec_elt<X: copy>(mv: myvec<X>) -> X { ret mv[0]; }

fn main() {
    let mv = myvec([1, 2, 3]);
    assert (myvec_deref(mv)[1] == 2);
    assert (myvec_elt(mv) == 1);
    assert (mv[2] == 3);
}
