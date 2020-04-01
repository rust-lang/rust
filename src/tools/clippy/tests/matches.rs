#![feature(rustc_private)]

extern crate rustc_span;
use std::collections::Bound;

#[test]
fn test_overlapping() {
    use clippy_lints::matches::overlapping;
    use rustc_span::source_map::DUMMY_SP;

    let sp = |s, e| clippy_lints::matches::SpannedRange {
        span: DUMMY_SP,
        node: (s, e),
    };

    assert_eq!(None, overlapping::<u8>(&[]));
    assert_eq!(None, overlapping(&[sp(1, Bound::Included(4))]));
    assert_eq!(
        None,
        overlapping(&[sp(1, Bound::Included(4)), sp(5, Bound::Included(6))])
    );
    assert_eq!(
        None,
        overlapping(&[
            sp(1, Bound::Included(4)),
            sp(5, Bound::Included(6)),
            sp(10, Bound::Included(11))
        ],)
    );
    assert_eq!(
        Some((&sp(1, Bound::Included(4)), &sp(3, Bound::Included(6)))),
        overlapping(&[sp(1, Bound::Included(4)), sp(3, Bound::Included(6))])
    );
    assert_eq!(
        Some((&sp(5, Bound::Included(6)), &sp(6, Bound::Included(11)))),
        overlapping(&[
            sp(1, Bound::Included(4)),
            sp(5, Bound::Included(6)),
            sp(6, Bound::Included(11))
        ],)
    );
}
