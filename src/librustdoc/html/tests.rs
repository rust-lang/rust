use rustc_span::{Symbol, create_default_session_globals_then, sym};

use crate::html::format::href_relative_parts;

fn assert_relative_path(expected: &str, relative_to_fqp: &[Symbol], fqp: &[Symbol]) {
    create_default_session_globals_then(|| {
        assert_eq!(expected, href_relative_parts(&fqp, &relative_to_fqp).finish());
    });
}

#[test]
fn href_relative_parts_basic() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::std, sym::iter];
    assert_relative_path("../iter", relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_parent_module() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::std];
    assert_relative_path("..", relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_different_crate() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::core, sym::iter];
    assert_relative_path("../../core/iter", relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_same_module() {
    let relative_to_fqp = &[sym::std, sym::vec];
    let fqp = &[sym::std, sym::vec];
    assert_relative_path("", relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_child_module() {
    let relative_to_fqp = &[sym::std];
    let fqp = &[sym::std, sym::vec];
    assert_relative_path("vec", relative_to_fqp, fqp);
}

#[test]
fn href_relative_parts_root() {
    let relative_to_fqp = &[];
    let fqp = &[sym::std];
    assert_relative_path("std", relative_to_fqp, fqp);
}
