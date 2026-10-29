//@ run-pass
fn main() {
    let mut t = [1; 2];
    t = [t[1] * 2, t[0] * 2];
    assert_eq!(&t[..], &[2, 2]);
}
