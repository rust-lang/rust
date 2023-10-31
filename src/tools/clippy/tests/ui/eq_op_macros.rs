#![warn(clippy::eq_op)]
#![allow(clippy::useless_vec)]

// lint also in macro definition
macro_rules! assert_in_macro_def {
    () => {
        let a = 42;
        assert_eq!(a, a);
        assert_ne!(a, a);
        debug_assert_eq!(a, a);
        debug_assert_ne!(a, a);
    };
}

// lint identical args in assert-like macro invocations (see #3574)
fn main() {
    assert_in_macro_def!();

    let a = 1;
    let b = 2;

    // lint identical args in `assert_eq!`
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

    let my_vec = vec![1; 5];
    let mut my_iter = my_vec.iter();
    assert_ne!(my_iter.next(), my_iter.next());
}
