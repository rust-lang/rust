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

    // lint identical args in `assert_ne!`
    assert_ne!(a, a);
    assert_ne!(a + 1, a + 1);
    // ok
    assert_ne!(a, b);
    assert_ne!(a, a + 1);
    assert_ne!(a + 1, b + 1);

    // lint identical args in `debug_assert_eq!`
    debug_assert_eq!(a, a);
    debug_assert_eq!(a + 1, a + 1);
    // ok
    debug_assert_eq!(a, b);
    debug_assert_eq!(a, a + 1);
    debug_assert_eq!(a + 1, b + 1);

    // lint identical args in `debug_assert_ne!`
    debug_assert_ne!(a, a);
    debug_assert_ne!(a + 1, a + 1);
    // ok
    debug_assert_ne!(a, b);
    debug_assert_ne!(a, a + 1);
    debug_assert_ne!(a + 1, b + 1);
}
