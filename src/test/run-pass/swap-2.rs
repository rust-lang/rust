fn swap[T](&vec[mutable T] v, int i, int j) {
    v.(i) <-> v.(j);
}

fn main() {
    let vec[mutable int] a = [mutable 0,1,2,3,4,5,6];
    swap(a, 2, 4);
    assert(a.(2) == 4);
    assert(a.(4) == 2);
    auto n = 42;
    n <-> a.(0);
    assert(a.(0) == 42);
    assert(n == 0);
}
