use super::*;

use crate::clean::collapse_doc_fragments;

use rustc_resolve::rustdoc::{unindent_doc_fragments, DocFragment, DocFragmentKind};
use rustc_span::create_default_session_globals_then;
use rustc_span::source_map::DUMMY_SP;
use rustc_span::symbol::Symbol;

fn create_doc_fragment(s: &str) -> Vec<DocFragment> {
    vec![DocFragment {
        span: DUMMY_SP,
        parent_module: None,
        doc: Symbol::intern(s),
        kind: DocFragmentKind::SugaredDoc,
        indent: 0,
    }]
}

#[track_caller]
fn run_test(input: &str, expected: &str) {
    create_default_session_globals_then(|| {
        let mut s = create_doc_fragment(input);
        unindent_doc_fragments(&mut s);
        assert_eq!(collapse_doc_fragments(&s), expected);
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
