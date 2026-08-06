// This file tests a bunch of tidy's own directives, so a lot of unwanted directives trigger here otherwise.
// By ignoring all, we can play around with whatever we'd like in this file
// ignore-tidy-file-all
use super::*;
use crate::diagnostics::TidyFlags;

#[test]
fn test_contains_problematic_const() {
    assert!(contains_problematic_const("721077")); // check with no "decimal" hex digits - converted to integer
    assert!(contains_problematic_const("524421")); // check with "decimal" replacements - converted to integer
    assert!(contains_problematic_const(&(285 * 281).to_string())); // check for hex display
    assert!(contains_problematic_const(&format!("{:x}B5", 2816))); // check for case-alternating hex display
    assert!(!contains_problematic_const("1193046")); // check for non-matching value
}

fn tidy_style_check(contents: &str) -> Vec<String> {
    let path = Path::new("/");
    let tidy_ctx = TidyCtx::new(path, false, None, TidyFlags::new(true));
    let mut check = tidy_ctx.start_check(CheckId::new("style").path(path));
    check_file_style(&mut check, &path.join("test.rs"), contents);

    check.get_errors()
}

#[test]
fn test_tidy_style_unused_directive() {
    assert_eq!(
        tidy_style_check(
            r#"// ignore-tidy-todo
"#
        ),
        vec!["/test.rs:1:  ignoring todo usage unnecessarily".to_string()]
    );
}

#[test]
fn test_tidy_style_unused_file_directive() {
    assert_eq!(
        tidy_style_check(
            r#"// ignore-tidy-file-todo
"#
        ),
        vec!["/test.rs:  ignoring todo usage unnecessarily".to_string()]
    );
}

#[test]
fn test_tidy_todo_usage_ignored_correctly() {
    assert_eq!(
        tidy_style_check(
            r#"// ignore-tidy-todo
todo!()
"#
        ),
        Vec::<String>::new()
    );
}

#[test]
fn test_tidy_todo_usage_ignored_correctly_file() {
    assert_eq!(
        tidy_style_check(
            r#"// ignore-tidy-file-todo
todo!()
"#
        ),
        Vec::<String>::new()
    );
}

#[test]
fn test_tidy_todo_usage_ignored_wrong_line() {
    assert_eq!(
        tidy_style_check(
            r#"// ignore-tidy-todo
foo
todo!()
"#
        ),
        vec![
            "/test.rs:1:  ignoring todo usage unnecessarily".to_string(),
            "/test.rs:3: the `todo!` macro is used for tasks that should be done before merging a PR. If you want to panic here, use `panic!`, `unimplemented!`, `unreachable!`, `rustc_middle::bug!` or an assertion".to_string()
        ]
    );
}

#[test]
fn test_tidy_todo_usage_ignored_correctly_file_line_inbetween() {
    assert_eq!(
        tidy_style_check(
            r#"// ignore-tidy-file-todo
foo
todo!()
"#
        ),
        Vec::<String>::new()
    );
}

#[test]
fn test_file_ignore_precedence() {
    assert_eq!(
        tidy_style_check(
            r#"// ignore-tidy-file-todo
// ignore-tidy-todo
todo!()
"#
        ),
        vec!["/test.rs:2:  ignoring todo usage unnecessarily".to_string()]
    );
}
