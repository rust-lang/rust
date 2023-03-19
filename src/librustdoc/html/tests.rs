use crate::html::format::href_relative_parts;
use rustc_span::{sym, Symbol};

fn assert_relative_path(expected: &[Symbol], relative_to_fqp: &[Symbol], fqp: &[Symbol]) {
    // No `create_default_session_globals_then` call is needed here because all
    // the symbols used are static, and no `Symbol::intern` calls occur.
    assert_eq!(expected, href_relative_parts(&fqp, &relative_to_fqp).collect::<Vec<_>>());
}

#[test]
fn href_relative_parts_basic() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::std, sym::iter];
    assert_relative_path(&[sym::dotdot, sym::iter], relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_parent_module() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::std];
    assert_relative_path(&[sym::dotdot], relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_different_crate() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::core, sym::iter];
    assert_relative_path(&[sym::dotdot, sym::dotdot, sym::core, sym::iter], relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_same_module() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::std, sym::vec];
    assert_relative_path(&[], relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_child_module() {
    let relative_to_fqp = &[sym::std];
    let fqp = &[sym::std, sym::vec];
    assert_relative_path(&[sym::vec], relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_root() {
    let relative_to_fqp = &[];
    let fqp = &[sym::std];
    assert_relative_path(&[sym::std], relative_to_fqp, fqp);
}

#[test]
fn test_html_remover() {
    use super::format::HtmlRemover;
    use std::fmt::Write;

    fn assert_removed_eq(input: &str, output: &str) {
        let mut remover = HtmlRemover::new(String::new());
        write!(&mut remover, "{}", input).unwrap();
        assert_eq!(&remover.into_inner(), output);
    }

    assert_removed_eq("a<a href='https://example.com'>b", "ab");
    assert_removed_eq("alpha &lt;bet&gt;", "alpha <bet>");
    assert_removed_eq("<a href=\"&quot;\">", "");
    assert_removed_eq("<tag>&gt;</tag>text&lt;<tag>", ">text<");
    assert_removed_eq("&quot;&#39;", "\"'");

    let mut remover = HtmlRemover::new(String::new());
    assert!(write!(&mut remover, "&ent;").is_err());

    let mut remover = HtmlRemover::new(String::new());
    assert!(write!(&mut remover, "&longentity").is_err());

    let mut remover = HtmlRemover::new(String::new());
    assert!(write!(&mut remover, "<open <tag").is_err());
}

#[test]
fn test_plain() {
    use super::format::Plain;
    let d = Plain("<strong>alpha</strong> &lt;bet&gt;");
    assert_eq!(&d.to_string(), "alpha <bet>");
}
