// This is a regression test for issue #135671 where a MIR refactor about arrays and their lengths
// unexpectedly caused borrowck errors for disjoint borrows of array elements, for which we had no
// tests. This is a collection of a few code samples from that issue.

//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

struct Test {
    a: i32,
    b: i32,
}

fn one() {
    let inputs: &mut [_] = &mut [Test { a: 0, b: 0 }];
    let a = &mut inputs[0].a;
    let b = &mut inputs[0].b;

    *a = 0;
    *b = 1;
}

fn two() {
    let slice = &mut [(0, 0)][..];
    std::mem::swap(&mut slice[0].0, &mut slice[0].1);
}

fn three(a: &mut [(i32, i32)], i: usize, j: usize) -> (&mut i32, &mut i32) {
    (&mut a[i].0, &mut a[j].1)
}

fn main() {
    one();
    two();
    three(&mut [(1, 2), (3, 4)], 0, 1);
}
