//@ run-pass

pub fn main() {
    let x = [22];
    let y = &x[0];
    assert_eq!(*y, 22);
}
