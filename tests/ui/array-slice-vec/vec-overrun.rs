//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    let v: Vec<isize> = vec![10];
    let x: usize = 0;
    assert_eq!(v[x], 10);
    // Bounds-check panic.

    assert_eq!(v[x + 2], 20);
}
