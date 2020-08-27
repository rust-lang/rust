use crate::ffi::OsString;
use crate::sys::windows::args::*;

fn chk(string: &str, parts: &[&str]) {
    let mut wide: Vec<u16> = OsString::from(string).encode_wide().collect();
    wide.push(0);
    let parsed =
        unsafe { parse_lp_cmd_line(wide.as_ptr() as *const u16, || OsString::from("TEST.EXE")) };
    let expected: Vec<OsString> = parts.iter().map(|k| OsString::from(k)).collect();
    assert_eq!(parsed.as_slice(), expected.as_slice());
}

#[test]
fn empty() {
    chk("", &["TEST.EXE"]);
    chk("\0", &["TEST.EXE"]);
}

#[test]
fn single_words() {
    chk("EXE one_word", &["EXE", "one_word"]);
    chk("EXE a", &["EXE", "a"]);
    chk("EXE 😅", &["EXE", "😅"]);
    chk("EXE 😅🤦", &["EXE", "😅🤦"]);
}

#[test]
fn official_examples() {
    chk(r#"EXE "abc" d e"#, &["EXE", "abc", "d", "e"]);
    chk(r#"EXE a\\\b d"e f"g h"#, &["EXE", r#"a\\\b"#, "de fg", "h"]);
    chk(r#"EXE a\\\"b c d"#, &["EXE", r#"a\"b"#, "c", "d"]);
    chk(r#"EXE a\\\\"b c" d e"#, &["EXE", r#"a\\b c"#, "d", "e"]);
}

#[test]
fn whitespace_behavior() {
    chk(r#" test"#, &["", "test"]);
    chk(r#"  test"#, &["", "test"]);
    chk(r#" test test2"#, &["", "test", "test2"]);
    chk(r#" test  test2"#, &["", "test", "test2"]);
    chk(r#"test test2 "#, &["test", "test2"]);
    chk(r#"test  test2 "#, &["test", "test2"]);
    chk(r#"test "#, &["test"]);
}

#[test]
fn genius_quotes() {
    chk(r#"EXE "" """#, &["EXE", "", ""]);
    chk(r#"EXE "" """"#, &["EXE", "", "\""]);
    chk(
        r#"EXE "this is """all""" in the same argument""#,
        &["EXE", "this is \"all\" in the same argument"],
    );
    chk(r#"EXE "a"""#, &["EXE", "a\""]);
    chk(r#"EXE "a"" a"#, &["EXE", "a\"", "a"]);
    // quotes cannot be escaped in command names
    chk(r#""EXE" check"#, &["EXE", "check"]);
    chk(r#""EXE check""#, &["EXE check"]);
    chk(r#""EXE """for""" check"#, &["EXE ", r#"for""#, "check"]);
    chk(r#""EXE \"for\" check"#, &[r#"EXE \"#, r#"for""#, "check"]);
}
