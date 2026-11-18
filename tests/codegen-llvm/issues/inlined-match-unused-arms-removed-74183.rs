//! Regression test for <https://github.com/rust-lang/rust/issues/74183>.
//!
//! After inlining `doc_comment_kind`, the match arms for block comments are
//! not needed by `is_line_doc_comment`, so the checks for `*` must be removed.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

enum AttrStyle {
    Inner,
    Outer,
}

enum DocCommentKind {
    Line,
    Block,
}

#[inline(always)]
const fn doc_comment_kind(s: &str) -> Option<(DocCommentKind, AttrStyle)> {
    match s.as_bytes() {
        [b'/', b'/', b'/', b'/', ..] => None,
        [b'/', b'/', b'/', ..] => Some((DocCommentKind::Line, AttrStyle::Inner)),
        [b'/', b'/', b'!', ..] => Some((DocCommentKind::Line, AttrStyle::Outer)),
        [b'/', b'*', b'*', b'*', _, ..] => None,
        [b'/', b'*', b'*', _, _, ..] => Some((DocCommentKind::Block, AttrStyle::Inner)),
        [b'/', b'*', b'!', _, _, ..] => Some((DocCommentKind::Block, AttrStyle::Outer)),
        _ => None,
    }
}

// CHECK-LABEL: @is_line_doc_comment
#[no_mangle]
pub const fn is_line_doc_comment(s: &str) -> bool {
    // CHECK-NOT: call
    // CHECK-NOT: i8 42
    // CHECK: ret i1
    match doc_comment_kind(s) {
        Some((DocCommentKind::Line, _)) => true,
        _ => false,
    }
}
