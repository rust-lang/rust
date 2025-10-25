#![allow(rustc::symbol_intern_string_literal)]

use rustc_span::{Symbol, create_default_session_globals_then};

use crate::builtin::is_hexagon_register_span;
use crate::levels::parse_lint_and_tool_name;

#[test]
fn parse_lint_no_tool() {
    create_default_session_globals_then(|| {
        assert_eq!(parse_lint_and_tool_name("foo"), (None, "foo"))
    });
}

#[test]
fn parse_lint_with_tool() {
    create_default_session_globals_then(|| {
        assert_eq!(parse_lint_and_tool_name("clippy::foo"), (Some(Symbol::intern("clippy")), "foo"))
    });
}

#[test]
fn parse_lint_multiple_path() {
    create_default_session_globals_then(|| {
        assert_eq!(
            parse_lint_and_tool_name("clippy::foo::bar"),
            (Some(Symbol::intern("clippy")), "foo::bar")
        )
    });
}

#[test]
fn test_hexagon_register_span_patterns() {
    // Valid Hexagon register span patterns
    assert!(is_hexagon_register_span("r1:0"));
    assert!(is_hexagon_register_span("r15:14"));
    assert!(is_hexagon_register_span("V5:4"));
    assert!(is_hexagon_register_span("V3:2"));
    assert!(is_hexagon_register_span("V5:4.w"));
    assert!(is_hexagon_register_span("V3:2.h"));
    assert!(is_hexagon_register_span("r99:98"));
    assert!(is_hexagon_register_span("V123:122.whatever"));

    // Invalid patterns - these should be treated as potential labels
    assert!(!is_hexagon_register_span("label1"));
    assert!(!is_hexagon_register_span("foo:"));
    assert!(!is_hexagon_register_span(":0"));
    assert!(!is_hexagon_register_span("r:0")); // missing digits before colon
    assert!(!is_hexagon_register_span("r1:")); // missing digits after colon
    assert!(!is_hexagon_register_span("r1:a")); // non-digit after colon
    assert!(!is_hexagon_register_span("1:0")); // starts with digit, not letter
    assert!(!is_hexagon_register_span("r1")); // no colon
    assert!(!is_hexagon_register_span("r")); // too short
    assert!(!is_hexagon_register_span("")); // empty
    assert!(!is_hexagon_register_span("ra:0")); // letter in first digit group
}
