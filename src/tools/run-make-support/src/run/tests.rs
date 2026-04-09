use std::ffi::OsString;

use super::split_maybe_args;

fn os(s: &str) -> OsString {
    OsString::from(s)
}

#[test]
fn split_on_space() {
    assert_eq!(
        split_maybe_args("valgrind --tool=memcheck"),
        vec![os("valgrind"), os("--tool=memcheck")]
    );
}

#[test]
fn single_arg_no_whitespace() {
    assert_eq!(split_maybe_args("valgrind"), vec![os("valgrind")]);
}

#[test]
fn empty_string() {
    assert_eq!(split_maybe_args(""), Vec::<OsString>::new());
}

#[test]
fn split_on_tab() {
    assert_eq!(
        split_maybe_args("valgrind\t--tool=memcheck"),
        vec![os("valgrind"), os("--tool=memcheck")]
    );
}

#[test]
fn split_on_newline() {
    assert_eq!(
        split_maybe_args("valgrind\n--tool=memcheck"),
        vec![os("valgrind"), os("--tool=memcheck")]
    );
}

#[test]
fn multiple_ifs_separators() {
    assert_eq!(split_maybe_args("a  b\t\tc\n\nd"), vec![os("a"), os("b"), os("c"), os("d")]);
}

#[test]
fn leading_and_trailing_whitespace() {
    assert_eq!(split_maybe_args("  valgrind\t"), vec![os("valgrind")]);
}
