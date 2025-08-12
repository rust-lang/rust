#![allow(unused, clippy::assertions_on_constants, clippy::const_is_empty)]
#![warn(clippy::bool_assert_comparison)]

use std::ops::{Add, Not};

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
#[derive(Debug, Clone, Copy)]
enum ImplNotTraitWithoutBool {
    VariantX(bool),
    VariantY(u32),
}

impl PartialEq<bool> for ImplNotTraitWithoutBool {
    fn eq(&self, other: &bool) -> bool {
        match *self {
            ImplNotTraitWithoutBool::VariantX(b) => b == *other,
            _ => false,
        }
    }
}

impl Not for ImplNotTraitWithoutBool {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            ImplNotTraitWithoutBool::VariantX(b) => ImplNotTraitWithoutBool::VariantX(!b),
            ImplNotTraitWithoutBool::VariantY(0) => ImplNotTraitWithoutBool::VariantY(1),
            ImplNotTraitWithoutBool::VariantY(_) => ImplNotTraitWithoutBool::VariantY(0),
        }
    }
}

// This type implements the Not trait with an Output of
// type bool. Using assert!(..) must be suggested
#[derive(Debug, Clone, Copy)]
struct ImplNotTraitWithBool;

impl PartialEq<bool> for ImplNotTraitWithBool {
    fn eq(&self, other: &bool) -> bool {
        false
    }
}

impl Not for ImplNotTraitWithBool {
    type Output = bool;

    fn not(self) -> Self::Output {
        true
    }
}

impl Add for ImplNotTraitWithBool {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self
    }
}

#[derive(Debug)]
struct NonCopy;

impl PartialEq<bool> for NonCopy {
    fn eq(&self, other: &bool) -> bool {
        false
    }
}

impl Not for NonCopy {
    type Output = bool;

    fn not(self) -> Self::Output {
        true
    }
}

fn main() {
    let a = ImplNotTraitWithoutBool::VariantX(true);
    let b = ImplNotTraitWithBool;

    assert_eq!("a".len(), 1);
    assert_eq!("a".is_empty(), false);
    //~^ bool_assert_comparison
    assert_eq!("".is_empty(), true);
    //~^ bool_assert_comparison
    assert_eq!(true, "".is_empty());
    //~^ bool_assert_comparison
    assert_eq!(a!(), b!());
    assert_eq!(a!(), "".is_empty());
    assert_eq!("".is_empty(), b!());
    assert_eq!(a, true);
    assert_eq!(b, true);
    //~^ bool_assert_comparison

    assert_ne!("a".len(), 1);
    assert_ne!("a".is_empty(), false);
    //~^ bool_assert_comparison
    assert_ne!("".is_empty(), true);
    //~^ bool_assert_comparison
    assert_ne!(true, "".is_empty());
    //~^ bool_assert_comparison
    assert_ne!(a!(), b!());
    assert_ne!(a!(), "".is_empty());
    assert_ne!("".is_empty(), b!());
    assert_ne!(a, true);
    assert_ne!(b, true);
    //~^ bool_assert_comparison

    debug_assert_eq!("a".len(), 1);
    debug_assert_eq!("a".is_empty(), false);
    //~^ bool_assert_comparison
    debug_assert_eq!("".is_empty(), true);
    //~^ bool_assert_comparison
    debug_assert_eq!(true, "".is_empty());
    //~^ bool_assert_comparison
    debug_assert_eq!(a!(), b!());
    debug_assert_eq!(a!(), "".is_empty());
    debug_assert_eq!("".is_empty(), b!());
    debug_assert_eq!(a, true);
    debug_assert_eq!(b, true);
    //~^ bool_assert_comparison

    debug_assert_ne!("a".len(), 1);
    debug_assert_ne!("a".is_empty(), false);
    //~^ bool_assert_comparison
    debug_assert_ne!("".is_empty(), true);
    //~^ bool_assert_comparison
    debug_assert_ne!(true, "".is_empty());
    //~^ bool_assert_comparison
    debug_assert_ne!(a!(), b!());
    debug_assert_ne!(a!(), "".is_empty());
    debug_assert_ne!("".is_empty(), b!());
    debug_assert_ne!(a, true);
    debug_assert_ne!(b, true);
    //~^ bool_assert_comparison

    // assert with error messages
    assert_eq!("a".len(), 1, "tadam {}", 1);
    assert_eq!("a".len(), 1, "tadam {}", true);
    assert_eq!("a".is_empty(), false, "tadam {}", 1);
    //~^ bool_assert_comparison
    assert_eq!("a".is_empty(), false, "tadam {}", true);
    //~^ bool_assert_comparison
    assert_eq!(false, "a".is_empty(), "tadam {}", true);
    //~^ bool_assert_comparison
    assert_eq!(a, true, "tadam {}", false);

    debug_assert_eq!("a".len(), 1, "tadam {}", 1);
    debug_assert_eq!("a".len(), 1, "tadam {}", true);
    debug_assert_eq!("a".is_empty(), false, "tadam {}", 1);
    //~^ bool_assert_comparison
    debug_assert_eq!("a".is_empty(), false, "tadam {}", true);
    //~^ bool_assert_comparison
    debug_assert_eq!(false, "a".is_empty(), "tadam {}", true);
    //~^ bool_assert_comparison
    debug_assert_eq!(a, true, "tadam {}", false);

    assert_eq!(a!(), true);
    //~^ bool_assert_comparison
    assert_eq!(true, b!());
    //~^ bool_assert_comparison

    use debug_assert_eq as renamed;
    renamed!(a, true);
    renamed!(b, true);
    //~^ bool_assert_comparison

    let non_copy = NonCopy;
    assert_eq!(non_copy, true);
    // changing the above to `assert!(non_copy)` would cause a `borrow of moved value`
    println!("{non_copy:?}");

    macro_rules! in_macro {
        ($v:expr) => {{
            assert_eq!($v, true);
        }};
    }
    in_macro!(a);

    assert_eq!("".is_empty(), true);
    //~^ bool_assert_comparison
    assert_ne!("".is_empty(), false);
    //~^ bool_assert_comparison
    assert_ne!("requires negation".is_empty(), true);
    //~^ bool_assert_comparison
    assert_eq!("requires negation".is_empty(), false);
    //~^ bool_assert_comparison

    debug_assert_eq!("".is_empty(), true);
    //~^ bool_assert_comparison
    debug_assert_ne!("".is_empty(), false);
    //~^ bool_assert_comparison
    debug_assert_ne!("requires negation".is_empty(), true);
    //~^ bool_assert_comparison
    debug_assert_eq!("requires negation".is_empty(), false);
    //~^ bool_assert_comparison
    assert_eq!(!b, true);
    //~^ bool_assert_comparison
    assert_eq!(!b, false);
    //~^ bool_assert_comparison
    assert_eq!(b + b, true);
    //~^ bool_assert_comparison
    assert_eq!(b + b, false);
    //~^ bool_assert_comparison
}
