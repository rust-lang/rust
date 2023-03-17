use crate::html::format::href_relative_parts;
use rustc_span::{sym, Symbol};

fn assert_relative_path(expected: (usize, &[Symbol]), relative_to_fqp: &[Symbol], fqp: &[Symbol]) {
    // No `create_default_session_globals_then` call is needed here because all
    // the symbols used are static, and no `Symbol::intern` calls occur.
    assert_eq!(expected, href_relative_parts(&fqp, &relative_to_fqp));
}

#[test]
fn href_relative_parts_basic() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::std, sym::iter];
    assert_relative_path((1, &[sym::iter]), relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_parent_module() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::std];
    assert_relative_path((1, &[]), relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_different_crate() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::core, sym::iter];
    assert_relative_path((2, &[sym::core, sym::iter]), relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_same_module() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::std, sym::vec];
    assert_relative_path((0, &[]), relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_child_module() {
    let relative_to_fqp = &[sym::std];
    let fqp = &[sym::std, sym::vec];
    assert_relative_path((0, &[sym::vec]), relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_root() {
    let relative_to_fqp = &[];
    let fqp = &[sym::std];
    assert_relative_path((0, &[sym::std]), relative_to_fqp, fqp);
}
