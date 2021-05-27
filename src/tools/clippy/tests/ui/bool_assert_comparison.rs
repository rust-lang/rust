#![warn(clippy::bool_assert_comparison)]

macro_rules! a {
    () => {
        true
    };
}
macro_rules! b {
    () => {
        true
    };
}

fn main() {
    assert_eq!("a".len(), 1);
    assert_eq!("a".is_empty(), false);
    assert_eq!("".is_empty(), true);
    assert_eq!(true, "".is_empty());
    assert_eq!(a!(), b!());
    assert_eq!(a!(), "".is_empty());
    assert_eq!("".is_empty(), b!());

    assert_ne!("a".len(), 1);
    assert_ne!("a".is_empty(), false);
    assert_ne!("".is_empty(), true);
    assert_ne!(true, "".is_empty());
    assert_ne!(a!(), b!());
    assert_ne!(a!(), "".is_empty());
    assert_ne!("".is_empty(), b!());

    debug_assert_eq!("a".len(), 1);
    debug_assert_eq!("a".is_empty(), false);
    debug_assert_eq!("".is_empty(), true);
    debug_assert_eq!(true, "".is_empty());
    debug_assert_eq!(a!(), b!());
    debug_assert_eq!(a!(), "".is_empty());
    debug_assert_eq!("".is_empty(), b!());

    debug_assert_ne!("a".len(), 1);
    debug_assert_ne!("a".is_empty(), false);
    debug_assert_ne!("".is_empty(), true);
    debug_assert_ne!(true, "".is_empty());
    debug_assert_ne!(a!(), b!());
    debug_assert_ne!(a!(), "".is_empty());
    debug_assert_ne!("".is_empty(), b!());

    // assert with error messages
    assert_eq!("a".len(), 1, "tadam {}", 1);
    assert_eq!("a".len(), 1, "tadam {}", true);
    assert_eq!("a".is_empty(), false, "tadam {}", 1);
    assert_eq!("a".is_empty(), false, "tadam {}", true);
    assert_eq!(false, "a".is_empty(), "tadam {}", true);

    debug_assert_eq!("a".len(), 1, "tadam {}", 1);
    debug_assert_eq!("a".len(), 1, "tadam {}", true);
    debug_assert_eq!("a".is_empty(), false, "tadam {}", 1);
    debug_assert_eq!("a".is_empty(), false, "tadam {}", true);
    debug_assert_eq!(false, "a".is_empty(), "tadam {}", true);
}
