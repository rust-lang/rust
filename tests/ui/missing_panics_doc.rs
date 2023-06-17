//@aux-build:macro_rules.rs
#![warn(clippy::missing_panics_doc)]
#![allow(clippy::option_map_unit_fn, clippy::unnecessary_literal_unwrap)]

#[macro_use]
extern crate macro_rules;

use macro_rules::macro_with_panic;

fn main() {}

/// This needs to be documented
pub fn unwrap() {
    let result = Err("Hi");
    result.unwrap()
}

/// This needs to be documented
pub fn panic() {
    panic!("This function panics")
}

/// This needs to be documented
pub fn inner_body(opt: Option<u32>) {
    opt.map(|x| {
        if x == 10 {
            panic!()
        }
    });
}

/// This needs to be documented
pub fn unreachable_and_panic() {
    if true { unreachable!() } else { panic!() }
}

/// This needs to be documented
pub fn assert_eq() {
    let x = 0;
    assert_eq!(x, 0);
}

/// This needs to be documented
pub fn assert_ne() {
    let x = 0;
    assert_ne!(x, 0);
}

/// This is documented
///
/// # Panics
///
/// Panics if `result` if an error
pub fn unwrap_documented() {
    let result = Err("Hi");
    result.unwrap()
}

/// This is documented
///
/// # Panics
///
/// Panics just because
pub fn panic_documented() {
    panic!("This function panics")
}

/// This is documented
///
/// # Panics
///
/// Panics if `opt` is Just(10)
pub fn inner_body_documented(opt: Option<u32>) {
    opt.map(|x| {
        if x == 10 {
            panic!()
        }
    });
}

/// This is documented
///
/// # Panics
///
/// We still need to do this part
pub fn unreachable_amd_panic_documented() {
    if true { unreachable!() } else { panic!() }
}

/// This is documented
///
/// # Panics
///
/// Panics if `x` is not 0.
pub fn assert_eq_documented() {
    let x = 0;
    assert_eq!(x, 0);
}

/// This is documented
///
/// # Panics
///
/// Panics if `x` is 0.
pub fn assert_ne_documented() {
    let x = 0;
    assert_ne!(x, 0);
}

/// `todo!()` is fine
pub fn todo() {
    todo!()
}

/// This is okay because it is private
fn unwrap_private() {
    let result = Err("Hi");
    result.unwrap()
}

/// This is okay because it is private
fn panic_private() {
    panic!("This function panics")
}

/// This is okay because it is private
fn inner_body_private(opt: Option<u32>) {
    opt.map(|x| {
        if x == 10 {
            panic!()
        }
    });
}

/// This is okay because unreachable
pub fn unreachable() {
    unreachable!("This function panics")
}

/// #6970.
/// This is okay because it is expansion of `debug_assert` family.
pub fn debug_assertions() {
    debug_assert!(false);
    debug_assert_eq!(1, 2);
    debug_assert_ne!(1, 2);
}

// all function must be triggered the lint.
// `pub` is required, because the lint does not consider unreachable items
pub mod issue10240 {
    pub fn option_unwrap<T>(v: &[T]) -> &T {
        let o: Option<&T> = v.last();
        o.unwrap()
    }

    pub fn option_expect<T>(v: &[T]) -> &T {
        let o: Option<&T> = v.last();
        o.expect("passed an empty thing")
    }

    pub fn result_unwrap<T>(v: &[T]) -> &T {
        let res: Result<&T, &str> = v.last().ok_or("oh noes");
        res.unwrap()
    }

    pub fn result_expect<T>(v: &[T]) -> &T {
        let res: Result<&T, &str> = v.last().ok_or("oh noes");
        res.expect("passed an empty thing")
    }

    pub fn last_unwrap(v: &[u32]) -> u32 {
        *v.last().unwrap()
    }

    pub fn last_expect(v: &[u32]) -> u32 {
        *v.last().expect("passed an empty thing")
    }
}

fn from_external_macro_should_not_lint() {
    macro_with_panic!()
}

macro_rules! some_macro_that_panics {
    () => {
        panic!()
    };
}

fn from_declared_macro_should_lint_at_macrosite() {
    // Not here.
    some_macro_that_panics!()
}
