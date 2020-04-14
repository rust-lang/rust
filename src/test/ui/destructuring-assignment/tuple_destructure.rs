// run-pass

#![feature(destructuring_assignment)]

fn main() {
    let (mut a, mut b);
    (a, b) = (0, 1);
    assert_eq!((a, b), (0, 1));
    (b, a) = (a, b);
    assert_eq!((a, b), (1, 0));
    (a, .., b) = (1, 2);
    assert_eq!((a, b), (1, 2));
    (_, a) = (1, 2);
    assert_eq!((a, b), (2, 2));
    (..) = (3, 4);
    assert_eq!((a, b), (2, 2));
}
