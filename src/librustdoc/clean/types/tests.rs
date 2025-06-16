use rustc_resolve::rustdoc::{DocFragmentKind, unindent_doc_fragments};
use rustc_span::create_default_session_globals_then;

use super::*;

fn create_doc_fragment(s: &str) -> Vec<DocFragment> {
    vec![DocFragment {
        span: DUMMY_SP,
        item_id: None,
        doc: Symbol::intern(s),
        kind: DocFragmentKind::SugaredDoc,
        indent: 0,
        from_expansion: false,
    }]
}

#[track_caller]
fn run_test(input: &str, expected: &str) {
    create_default_session_globals_then(|| {
        let mut s = create_doc_fragment(input);
        unindent_doc_fragments(&mut s);
        let attrs = Attributes { doc_strings: s, other_attrs: Default::default() };
        assert_eq!(attrs.doc_value(), expected);
    });
}

#[test]
fn should_unindent() {
    run_test("    line1\n    line2", "line1\nline2");
}

#[test]
fn should_unindent_multiple_paragraphs() {
    run_test("    line1\n\n    line2", "line1\n\nline2");
}

#[test]
fn should_leave_multiple_indent_levels() {
    // Line 2 is indented another level beyond the
    // base indentation and should be preserved
    run_test("    line1\n\n        line2", "line1\n\n    line2");
}

#[test]
fn should_ignore_first_line_indent() {
    run_test("line1\n    line2", "line1\n    line2");
}

#[test]
fn should_not_ignore_first_line_indent_in_a_single_line_para() {
    run_test("line1\n\n    line2", "line1\n\n    line2");
}

#[test]
fn should_unindent_tabs() {
    run_test("\tline1\n\tline2", "line1\nline2");
}

#[test]
fn should_trim_mixed_indentation() {
    run_test("\t    line1\n\t    line2", "line1\nline2");
    run_test("    \tline1\n    \tline2", "line1\nline2");
}

#[test]
fn should_not_trim() {
    run_test("\t    line1  \n\t    line2", "line1  \nline2");
    run_test("    \tline1  \n    \tline2", "line1  \nline2");
}

#[test]
fn is_same_generic() {
    use crate::clean::types::{PrimitiveType, Type};
    use crate::formats::cache::Cache;
    let cache = Cache::new(false, false);
    let generic = Type::Generic(rustc_span::symbol::sym::Any);
    let unit = Type::Primitive(PrimitiveType::Unit);
    assert!(!generic.is_doc_subtype_of(&unit, &cache));
    assert!(unit.is_doc_subtype_of(&generic, &cache));
}
