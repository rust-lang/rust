//! Parser precedence test to help with [RFC 87 "Trait Bounds with Plus"][rfc-87], to check the
//! precedence of the `as` operator in relation to some arithmetic bin-ops and parentheses.
//!
//! Editor's note: this test seems quite incomplete compared to what's possible nowadays. Maybe
//! there's another set of tests whose coverage overshadows this test?
//!
//! [rfc-87]: https://rust-lang.github.io/rfcs/0087-trait-bounds-with-plus.html

//@ run-pass

#[allow(unused_parens)]
fn main() {
    assert_eq!(3 as usize * 3, 9);
    assert_eq!(3 as (usize) * 3, 9);
    assert_eq!(3 as (usize) / 3, 1);
    assert_eq!(3 as usize + 3, 6);
    assert_eq!(3 as (usize) + 3, 6);
}
