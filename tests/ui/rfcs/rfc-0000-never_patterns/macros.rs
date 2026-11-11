//@ check-pass
//@ revisions: e2018 e2021
//@[e2018] edition:2018
//@[e2021] edition:2021
#![feature(never_patterns)]
#![allow(incomplete_features)]

#[derive(Debug, PartialEq, Eq)]
struct Pattern;
#[derive(Debug, PartialEq, Eq)]
struct Never;
#[derive(Debug, PartialEq, Eq)]
struct Other;

macro_rules! detect_pat {
    ($p:pat) => {
        Pattern
    };
    (!) => {
        Never
    };
    ($($x:tt)*) => {
        Other
    };
}

// For backwards-compatibility, all the cases that parse as `Pattern` under the feature gate must
// have been parse errors before.
fn main() {
    // For backwards compatibility this does not match `$p:pat`.
    assert_eq!(detect_pat!(!), Never);

    // Edition 2018 parses both of these cases as `Other`. Both editions have been parsing the
    // first case as `Other` before, so we mustn't change that.
    assert_eq!(detect_pat!(! | true), Other);
    #[cfg(e2018)]
    assert_eq!(detect_pat!(true | !), Other);
    #[cfg(e2021)]
    assert_eq!(detect_pat!(true | !), Pattern);

    // These are never patterns; they take no body when they're in a match arm.
    assert_eq!(detect_pat!((!)), Pattern);
    assert_eq!(detect_pat!((true, !)), Pattern);
    assert_eq!(detect_pat!(Some(!)), Pattern);

    // These count as normal patterns.
    assert_eq!(detect_pat!((! | true)), Pattern);
    assert_eq!(detect_pat!((Ok(x) | Err(&!))), Pattern);
}
