//@ run-pass
#![allow(dead_code)]
struct S { a: usize }

static A: S = S { a: 3 };
static B: &'static usize = &A.a;
static C: &'static usize = &(A.a);

static D: [usize; 1] = [1];
static E: usize = D[0];
static F: &'static usize = &D[0];

fn main() {
    assert_eq!(*B, A.a);
    assert_eq!(*B, A.a);

    assert_eq!(E, D[0]);
    assert_eq!(*F, D[0]);
}
