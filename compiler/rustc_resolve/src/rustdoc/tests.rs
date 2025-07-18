use std::path::PathBuf;

use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::symbol::sym;
use rustc_span::{BytePos, Span};

use super::{DocFragment, DocFragmentKind, source_span_for_markdown_range_inner};

#[test]
fn single_backtick() {
    let sm = SourceMap::new(FilePathMapping::empty());
    sm.new_source_file(PathBuf::from("foo.rs").into(), r#"#[doc = "`"] fn foo() {}"#.to_string());
    let (span, _) = source_span_for_markdown_range_inner(
        &sm,
        "`",
        &(0..1),
        &[DocFragment {
            span: Span::with_root_ctxt(BytePos(8), BytePos(11)),
            item_id: None,
            kind: DocFragmentKind::RawDoc,
            doc: sym::empty, // unused placeholder
            indent: 0,
            from_expansion: false,
        }],
    )
    .unwrap();
    assert_eq!(span.lo(), BytePos(9));
    assert_eq!(span.hi(), BytePos(10));
}

#[test]
fn utf8() {
    // regression test for https://github.com/rust-lang/rust/issues/141665
    let sm = SourceMap::new(FilePathMapping::empty());
    sm.new_source_file(PathBuf::from("foo.rs").into(), r#"#[doc = "⚠"] fn foo() {}"#.to_string());
    let (span, _) = source_span_for_markdown_range_inner(
        &sm,
        "⚠",
        &(0..3),
        &[DocFragment {
            span: Span::with_root_ctxt(BytePos(8), BytePos(14)),
            item_id: None,
            kind: DocFragmentKind::RawDoc,
            doc: sym::empty, // unused placeholder
            indent: 0,
            from_expansion: false,
        }],
    )
    .unwrap();
    assert_eq!(span.lo(), BytePos(9));
    assert_eq!(span.hi(), BytePos(12));
}
