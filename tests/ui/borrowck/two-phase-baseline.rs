//@ run-pass

// This is the "goto example" for why we want two phase borrows.

fn main() {
    let mut v = vec![0, 1, 2];
    v.push(v.len());
    assert_eq!(v, [0, 1, 2, 3]);
}
