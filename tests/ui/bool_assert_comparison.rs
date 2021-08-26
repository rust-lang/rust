#![warn(clippy::bool_assert_comparison)]

use std::ops::Not;

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

// Implements the Not trait but with an output type
// that's not bool. Should not suggest a rewrite
#[derive(Debug)]
enum A {
    VariantX(bool),
    VariantY(u32),
}

impl PartialEq<bool> for A {
    fn eq(&self, other: &bool) -> bool {
        match *self {
            A::VariantX(b) => b == *other,
            _ => false,
        }
    }
}

impl Not for A {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            A::VariantX(b) => A::VariantX(!b),
            A::VariantY(0) => A::VariantY(1),
            A::VariantY(_) => A::VariantY(0),
        }
    }
}

// This type implements the Not trait with an Output of
// type bool. Using assert!(..) must be suggested
#[derive(Debug)]
struct B;

impl PartialEq<bool> for B {
    fn eq(&self, other: &bool) -> bool {
        false
    }
}

impl Not for B {
    type Output = bool;

    fn not(self) -> Self::Output {
        true
    }
}

fn main() {
    let a = A::VariantX(true);
    let b = B {};

    assert_eq!("a".len(), 1);
    assert_eq!("a".is_empty(), false);
    assert_eq!("".is_empty(), true);
    assert_eq!(true, "".is_empty());
    assert_eq!(a!(), b!());
    assert_eq!(a!(), "".is_empty());
    assert_eq!("".is_empty(), b!());
    assert_eq!(a, true);
    assert_eq!(b, true);

    assert_ne!("a".len(), 1);
    assert_ne!("a".is_empty(), false);
    assert_ne!("".is_empty(), true);
    assert_ne!(true, "".is_empty());
    assert_ne!(a!(), b!());
    assert_ne!(a!(), "".is_empty());
    assert_ne!("".is_empty(), b!());
    assert_ne!(a, true);
    assert_ne!(b, true);

    debug_assert_eq!("a".len(), 1);
    debug_assert_eq!("a".is_empty(), false);
    debug_assert_eq!("".is_empty(), true);
    debug_assert_eq!(true, "".is_empty());
    debug_assert_eq!(a!(), b!());
    debug_assert_eq!(a!(), "".is_empty());
    debug_assert_eq!("".is_empty(), b!());
    debug_assert_eq!(a, true);
    debug_assert_eq!(b, true);

    debug_assert_ne!("a".len(), 1);
    debug_assert_ne!("a".is_empty(), false);
    debug_assert_ne!("".is_empty(), true);
    debug_assert_ne!(true, "".is_empty());
    debug_assert_ne!(a!(), b!());
    debug_assert_ne!(a!(), "".is_empty());
    debug_assert_ne!("".is_empty(), b!());
    debug_assert_ne!(a, true);
    debug_assert_ne!(b, true);

    // assert with error messages
    assert_eq!("a".len(), 1, "tadam {}", 1);
    assert_eq!("a".len(), 1, "tadam {}", true);
    assert_eq!("a".is_empty(), false, "tadam {}", 1);
    assert_eq!("a".is_empty(), false, "tadam {}", true);
    assert_eq!(false, "a".is_empty(), "tadam {}", true);
    assert_eq!(a, true, "tadam {}", false);

    debug_assert_eq!("a".len(), 1, "tadam {}", 1);
    debug_assert_eq!("a".len(), 1, "tadam {}", true);
    debug_assert_eq!("a".is_empty(), false, "tadam {}", 1);
    debug_assert_eq!("a".is_empty(), false, "tadam {}", true);
    debug_assert_eq!(false, "a".is_empty(), "tadam {}", true);
    debug_assert_eq!(a, true, "tadam {}", false);
}
