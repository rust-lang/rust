#![warn(clippy::inline_always, clippy::deprecated_semver)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::missing_docs_in_private_items, clippy::panic, clippy::unreachable)]

#[inline(always)]
//~^ ERROR: you have declared `#[inline(always)]` on `test_attr_lint`. This is usually a b
//~| NOTE: `-D clippy::inline-always` implied by `-D warnings`
fn test_attr_lint() {
    assert!(true)
}

#[inline(always)]
fn false_positive_expr() {
    unreachable!()
}

#[inline(always)]
fn false_positive_stmt() {
    unreachable!();
}

#[inline(always)]
fn empty_and_false_positive_stmt() {
    unreachable!();
}

#[deprecated(since = "forever")]
//~^ ERROR: the since field must contain a semver-compliant version
//~| NOTE: `-D clippy::deprecated-semver` implied by `-D warnings`
pub const SOME_CONST: u8 = 42;

#[deprecated(since = "1")]
//~^ ERROR: the since field must contain a semver-compliant version
pub const ANOTHER_CONST: u8 = 23;

#[deprecated(since = "0.1.1")]
pub const YET_ANOTHER_CONST: u8 = 0;

fn main() {
    test_attr_lint();
    if false {
        false_positive_expr()
    }
    if false {
        false_positive_stmt()
    }
    if false {
        empty_and_false_positive_stmt()
    }
}
