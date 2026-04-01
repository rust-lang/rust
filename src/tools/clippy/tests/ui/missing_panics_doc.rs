//@aux-build:macro_rules.rs
#![warn(clippy::missing_panics_doc)]
#![allow(clippy::option_map_unit_fn, clippy::unnecessary_literal_unwrap)]

#[macro_use]
extern crate macro_rules;

use macro_rules::macro_with_panic;

fn main() {}

/// This needs to be documented
pub fn unwrap() {
    //~^ missing_panics_doc
    let result = Err("Hi");
    result.unwrap()
}

/// This needs to be documented
pub fn panic() {
    //~^ missing_panics_doc
    panic!("This function panics")
}

/// This needs to be documented
pub fn inner_body(opt: Option<u32>) {
    //~^ missing_panics_doc
    opt.map(|x| {
        if x == 10 {
            panic!()
        }
    });
}

/// This needs to be documented
pub fn unreachable_and_panic() {
    //~^ missing_panics_doc
    if true { unreachable!() } else { panic!() }
}

/// This needs to be documented
pub fn assert_eq() {
    //~^ missing_panics_doc
    let x = 0;
    assert_eq!(x, 0);
}

/// This needs to be documented
pub fn assert_ne() {
    //~^ missing_panics_doc
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

pub fn partially_const<const N: usize>(n: usize) {
    //~^ missing_panics_doc

    const {
        assert!(N > 5);
    }

    assert!(N > n);
}

pub fn expect_allow(i: Option<isize>) {
    #[expect(clippy::missing_panics_doc)]
    i.unwrap();

    #[allow(clippy::missing_panics_doc)]
    i.unwrap();
}

pub fn expect_allow_with_error(i: Option<isize>) {
    //~^ missing_panics_doc

    #[expect(clippy::missing_panics_doc)]
    i.unwrap();

    #[allow(clippy::missing_panics_doc)]
    i.unwrap();

    i.unwrap();
}

pub fn expect_after_error(x: Option<u32>, y: Option<u32>) {
    //~^ missing_panics_doc

    let x = x.unwrap();

    #[expect(clippy::missing_panics_doc)]
    let y = y.unwrap();
}

// all function must be triggered the lint.
// `pub` is required, because the lint does not consider unreachable items
pub mod issue10240 {
    pub fn option_unwrap<T>(v: &[T]) -> &T {
        //~^ missing_panics_doc
        let o: Option<&T> = v.last();
        o.unwrap()
    }

    pub fn option_expect<T>(v: &[T]) -> &T {
        //~^ missing_panics_doc
        let o: Option<&T> = v.last();
        o.expect("passed an empty thing")
    }

    pub fn result_unwrap<T>(v: &[T]) -> &T {
        //~^ missing_panics_doc
        let res: Result<&T, &str> = v.last().ok_or("oh noes");
        res.unwrap()
    }

    pub fn result_expect<T>(v: &[T]) -> &T {
        //~^ missing_panics_doc
        let res: Result<&T, &str> = v.last().ok_or("oh noes");
        res.expect("passed an empty thing")
    }

    pub fn last_unwrap(v: &[u32]) -> u32 {
        //~^ missing_panics_doc
        *v.last().unwrap()
    }

    pub fn last_expect(v: &[u32]) -> u32 {
        //~^ missing_panics_doc
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

pub fn issue_12760<const N: usize>() {
    const {
        if N == 0 {
            panic!();
        }
    }
}

/// This needs documenting
pub fn unwrap_expect_etc_in_const() {
    let a = const { std::num::NonZeroUsize::new(1).unwrap() };
    // This should still pass the lint even if it is guaranteed to panic at compile-time
    let b = const { std::num::NonZeroUsize::new(0).unwrap() };
}

/// This needs documenting
pub const fn unwrap_expect_etc_in_const_fn_fails() {
    //~^ missing_panics_doc
    let a = std::num::NonZeroUsize::new(1).unwrap();
}

/// This needs documenting
pub const fn assert_in_const_fn_fails() {
    //~^ missing_panics_doc
    let x = 0;
    if x == 0 {
        panic!();
    }
}

/// This needs documenting
pub const fn in_const_fn<const N: usize>(n: usize) {
    //~^ missing_panics_doc
    assert!(N > n);
}
