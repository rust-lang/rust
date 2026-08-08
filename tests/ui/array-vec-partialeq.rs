//@ check-pass

fn main() {
    let x = vec![1];
    let y = [1];
    assert!(x == y);
    assert!(y == x);

    let z = [1, 2, 3];
    assert!(z == vec![1, 2, 3]);
    assert!(&z == vec![1, 2, 3]);
}
