//@ compile-flags: -Zmir-opt-level=3 -Copt-level=0
//@ run-pass

type M = [i64; 2];

fn f(a: &M) -> M {
    let mut b: M = M::default();
    b[0] = a[0] * a[0];
    b
}

fn main() {
    let mut a: M = [1, 1];
    a = f(&a);
    assert_eq!(a[0], 1);
}
