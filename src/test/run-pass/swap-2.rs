fn swap<T>(v: [mut T], i: int, j: int) { v[i] <-> v[j]; }

fn main() {
    let a: [mut int]/~ = [mut 0, 1, 2, 3, 4, 5, 6]/~;
    swap(a, 2, 4);
    assert (a[2] == 4);
    assert (a[4] == 2);
    let mut n = 42;
    n <-> a[0];
    assert (a[0] == 42);
    assert (n == 0);
}
