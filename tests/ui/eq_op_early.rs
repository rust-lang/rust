#![warn(clippy::eq_op)]

fn main() {
    let a = 1;
    let b = 2;

    // lint identical args in `assert_eq!` (see #3574)
    assert_eq!(a, a);
    assert_eq!(a + 1, a + 1);

    // ok
    assert_eq!(a, b);
    assert_eq!(a, a + 1);
    assert_eq!(a + 1, b + 1);
}
