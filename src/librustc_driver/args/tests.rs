use super::*;

use std::str;

fn want_args(v: impl IntoIterator<Item = &'static str>) -> Vec<String> {
    v.into_iter().map(String::from).collect()
}

fn got_args(file: &[u8]) -> Result<Vec<String>, Error> {
    let ret = str::from_utf8(file)
        .map_err(|_| Error::Utf8Error(None))?
        .lines()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    Ok(ret)
}

#[test]
fn nothing() {
    let file = b"";

    assert_eq!(got_args(file).unwrap(), want_args(vec![]));
}

#[test]
fn empty() {
    let file = b"\n";

    assert_eq!(got_args(file).unwrap(), want_args(vec![""]));
}

#[test]
fn simple() {
    let file = b"foo";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo"]));
}

#[test]
fn simple_eol() {
    let file = b"foo\n";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo"]));
}

#[test]
fn multi() {
    let file = b"foo\nbar";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "bar"]));
}

#[test]
fn multi_eol() {
    let file = b"foo\nbar\n";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "bar"]));
}

#[test]
fn multi_empty() {
    let file = b"foo\n\nbar";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "", "bar"]));
}

#[test]
fn multi_empty_eol() {
    let file = b"foo\n\nbar\n";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "", "bar"]));
}

#[test]
fn multi_empty_start() {
    let file = b"\nfoo\nbar";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["", "foo", "bar"]));
}

#[test]
fn multi_empty_end() {
    let file = b"foo\nbar\n\n";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "bar", ""]));
}

#[test]
fn simple_eol_crlf() {
    let file = b"foo\r\n";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo"]));
}

#[test]
fn multi_crlf() {
    let file = b"foo\r\nbar";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "bar"]));
}

#[test]
fn multi_eol_crlf() {
    let file = b"foo\r\nbar\r\n";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "bar"]));
}

#[test]
fn multi_empty_crlf() {
    let file = b"foo\r\n\r\nbar";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "", "bar"]));
}

#[test]
fn multi_empty_eol_crlf() {
    let file = b"foo\r\n\r\nbar\r\n";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "", "bar"]));
}

#[test]
fn multi_empty_start_crlf() {
    let file = b"\r\nfoo\r\nbar";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["", "foo", "bar"]));
}

#[test]
fn multi_empty_end_crlf() {
    let file = b"foo\r\nbar\r\n\r\n";

    assert_eq!(got_args(file).unwrap(), want_args(vec!["foo", "bar", ""]));
}

#[test]
fn bad_utf8() {
    let file = b"foo\x80foo";

    match got_args(file).unwrap_err() {
        Error::Utf8Error(_) => (),
        bad => panic!("bad err: {:?}", bad),
    }
}
