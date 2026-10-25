//! Regression test for https://github.com/rust-lang/rust/issues/148423.
//!
//! `derive(PartialEq)` used to emit `core::intrinsics::unreachable()` as the
//! default arm of the generated `match`. A non-inert attribute macro applied
//! after the derive can remove or alter variants after the derive has seen
//! them, which makes that arm reachable and turns the intrinsic into undefined
//! behavior in safe code.
//!
//! The `discard` proc macro below drops the enum definition that the derives
//! saw; the real `Thing` declared afterwards has an extra `Three` variant,
//! which only the default arm of the derived code can handle. `PartialOrd` and
//! `Ord` are derived as well to cover their (already sound) discriminant
//! fallbacks.

//@ run-pass
//@ proc-macro: partial-eq-discard-macro-issue-148423.rs

use std::cmp::Ordering;

extern crate partial_eq_discard_macro_issue_148423 as discard;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[discard::discard]
enum Thing {
    One(i32),
    Two(i32),
}

// The `Three` field is never read: only the discriminant of the extra variant
// is compared.
#[allow(dead_code)]
enum Thing {
    One(i32),
    Two(i32),
    Three(u64),
}

fn main() {
    // The extra variant falls back to a discriminant comparison, like a
    // fieldless enum: same variant => equal, even if the fields differ.
    assert!(Thing::Three(1) == Thing::Three(1));
    assert!(Thing::Three(1) == Thing::Three(2));
    assert!(Thing::One(1).partial_cmp(&Thing::Three(2)) == Some(Ordering::Less));
    assert!(Thing::Three(1).cmp(&Thing::Three(2)) == Ordering::Equal);
    // Variants known to the derives keep comparing their fields.
    assert!(Thing::One(1) != Thing::One(2));
    assert!(Thing::One(1) != Thing::Two(1));
    assert!(Thing::One(1).partial_cmp(&Thing::One(2)) == Some(Ordering::Less));
    assert!(Thing::One(1).partial_cmp(&Thing::Two(1)) == Some(Ordering::Less));
    assert!(Thing::One(1).cmp(&Thing::One(2)) == Ordering::Less);
    assert!(Thing::One(1).cmp(&Thing::Two(1)) == Ordering::Less);
}
