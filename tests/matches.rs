#![allow(plugin_as_library)]
#![feature(rustc_private)]

extern crate clippy;
extern crate syntax;

#[test]
fn test_overlapping() {
    use clippy::matches::overlapping;
    use syntax::codemap::DUMMY_SP;

    let sp = |s, e| {
        clippy::matches::SpannedRange {
            span: DUMMY_SP,
            node: (s, e),
        }
    };

    assert_eq!(None, overlapping::<u8>(&[]));
    assert_eq!(None, overlapping(&[sp(1, 4)]));
    assert_eq!(None, overlapping(&[sp(1, 4), sp(5, 6)]));
    assert_eq!(None, overlapping(&[sp(1, 4), sp(5, 6), sp(10, 11)]));
    assert_eq!(Some((&sp(1, 4), &sp(3, 6))), overlapping(&[sp(1, 4), sp(3, 6)]));
    assert_eq!(Some((&sp(5, 6), &sp(6, 11))), overlapping(&[sp(1, 4), sp(5, 6), sp(6, 11)]));
}
