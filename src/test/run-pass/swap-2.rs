// xfail-pretty

fn swap[@T](v: &vec[mutable T], i: int, j: int) { v.(i) <-> v.(j); }

fn main() {
    let a: vec[mutable int] = [mutable 0, 1, 2, 3, 4, 5, 6];
    swap(a, 2, 4);
    assert (a.(2) == 4);
    assert (a.(4) == 2);
    let n = 42;
    n <-> a.(0);
    assert (a.(0) == 42);
    assert (n == 0);
}