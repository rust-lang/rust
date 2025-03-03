#![warn(clippy::inline_always, clippy::deprecated_semver)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::missing_docs_in_private_items, clippy::panic, clippy::unreachable)]

#[inline(always)]
//~^ inline_always
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
//~^ deprecated_semver
pub const SOME_CONST: u8 = 42;

#[deprecated(since = "1")]
//~^ deprecated_semver
pub const ANOTHER_CONST: u8 = 23;

#[deprecated(since = "0.1.1")]
pub const YET_ANOTHER_CONST: u8 = 0;

#[deprecated(since = "TBD")]
pub const GONNA_DEPRECATE_THIS_LATER: u8 = 0;

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
