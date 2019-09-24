use super::*;

#[test]
fn should_unindent() {
    let s = "    line1\n    line2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1\nline2");
}

#[test]
fn should_unindent_multiple_paragraphs() {
    let s = "    line1\n\n    line2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1\n\nline2");
}

#[test]
fn should_leave_multiple_indent_levels() {
    // Line 2 is indented another level beyond the
    // base indentation and should be preserved
    let s = "    line1\n\n        line2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1\n\n    line2");
}

#[test]
fn should_ignore_first_line_indent() {
    // The first line of the first paragraph may not be indented as
    // far due to the way the doc string was written:
    //
    // #[doc = "Start way over here
    //          and continue here"]
    let s = "line1\n    line2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1\nline2");
}

#[test]
fn should_not_ignore_first_line_indent_in_a_single_line_para() {
    let s = "line1\n\n    line2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1\n\n    line2");
}

#[test]
fn should_unindent_tabs() {
    let s = "\tline1\n\tline2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1\nline2");
}

#[test]
fn should_trim_mixed_indentation() {
    let s = "\t    line1\n\t    line2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1\nline2");

    let s = "    \tline1\n    \tline2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1\nline2");
}

#[test]
fn should_not_trim() {
    let s = "\t    line1  \n\t    line2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1  \nline2");

    let s = "    \tline1  \n    \tline2".to_string();
    let r = unindent(&s);
    assert_eq!(r, "line1  \nline2");
}
