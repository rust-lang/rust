tag myvec[X] = vec[X];

fn myvec_deref[X](&myvec[X] mv) -> vec[X] {
    ret *mv;
}

fn myvec_elt[X](&myvec[X] mv) -> X {
    ret mv.(0);
}

fn main() {
    auto mv = myvec([1, 2, 3]);
    assert(myvec_deref(mv).(1) == 2);
    assert(myvec_elt(mv) == 1);
    assert(mv.(2) == 3);
}
