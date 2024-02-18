//@ run-pass

pub fn main() {
    let x = 1;
    let y = 1;
    assert_eq!(&x, &y);
}
