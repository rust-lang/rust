// run-pass

fn main() {
    let (mut a, mut b);
    (a, b) = (0, 1);
    assert_eq!((a, b), (0, 1));
    (b, a) = (a, b);
    assert_eq!((a, b), (1, 0));
    (a, .., b) = (1, 2);
    assert_eq!((a, b), (1, 2));
    (.., a) = (1, 2);
    assert_eq!((a, b), (2, 2));
    (..) = (3, 4);
    assert_eq!((a, b), (2, 2));
    (b, ..) = (5, 6, 7);
    assert_eq!(b, 5);
    (a, _) = (8, 9);
    assert_eq!(a, 8);

    // Test for a non-Copy type (String):
    let (mut c, mut d);
    (c, d) = ("c".to_owned(), "d".to_owned());
    assert_eq!(c, "c");
    assert_eq!(d, "d");
    (d, c) = (c, d);
    assert_eq!(c, "d");
    assert_eq!(d, "c");

    // Test nesting/parentheses:
    ((a, b)) = (0, 1);
    assert_eq!((a, b), (0, 1));
    (((a, b)), (c)) = ((2, 3), d);
    assert_eq!((a, b), (2, 3));
    assert_eq!(c, "c");
    ((a, .., b), .., (..)) = ((4, 5), ());
    assert_eq!((a, b), (4, 5));
}
