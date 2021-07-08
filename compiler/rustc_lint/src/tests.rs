use crate::context::parse_lint_and_tool_name;
use rustc_span::{with_default_session_globals, Symbol};

#[test]
fn parse_lint_no_tool() {
    with_default_session_globals(|| assert_eq!(parse_lint_and_tool_name("foo"), (None, "foo")));
}

#[test]
fn parse_lint_with_tool() {
    with_default_session_globals(|| {
        assert_eq!(parse_lint_and_tool_name("clippy::foo"), (Some(Symbol::intern("clippy")), "foo"))
    });
}

#[test]
fn parse_lint_multiple_path() {
    with_default_session_globals(|| {
        assert_eq!(
            parse_lint_and_tool_name("clippy::foo::bar"),
            (Some(Symbol::intern("clippy")), "foo::bar")
        )
    });
}
