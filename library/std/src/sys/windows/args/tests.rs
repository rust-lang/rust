use crate::ffi::OsString;
use crate::sys::windows::args::*;

fn chk(string: &str, parts: &[&str]) {
    let mut wide: Vec<u16> = OsString::from(string).encode_wide().collect();
    wide.push(0);
    let parsed =
        unsafe { parse_lp_cmd_line(WStrUnits::new(wide.as_ptr()), || OsString::from("TEST.EXE")) };
    let expected: Vec<OsString> = parts.iter().map(|k| OsString::from(k)).collect();
    assert_eq!(parsed.as_slice(), expected.as_slice(), "{:?}", string);
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
    chk(r#"EXE a\\\b d"e f"g h"#, &["EXE", r"a\\\b", "de fg", "h"]);
    chk(r#"EXE a\\\"b c d"#, &["EXE", r#"a\"b"#, "c", "d"]);
    chk(r#"EXE a\\\\"b c" d e"#, &["EXE", r"a\\b c", "d", "e"]);
}

#[test]
fn whitespace_behavior() {
    chk(" test", &["", "test"]);
    chk("  test", &["", "test"]);
    chk(" test test2", &["", "test", "test2"]);
    chk(" test  test2", &["", "test", "test2"]);
    chk("test test2 ", &["test", "test2"]);
    chk("test  test2 ", &["test", "test2"]);
    chk("test ", &["test"]);
}

#[test]
fn genius_quotes() {
    chk(r#"EXE "" """#, &["EXE", "", ""]);
    chk(r#"EXE "" """"#, &["EXE", "", r#"""#]);
    chk(
        r#"EXE "this is """all""" in the same argument""#,
        &["EXE", r#"this is "all" in the same argument"#],
    );
    chk(r#"EXE "a"""#, &["EXE", r#"a""#]);
    chk(r#"EXE "a"" a"#, &["EXE", r#"a" a"#]);
    // quotes cannot be escaped in command names
    chk(r#""EXE" check"#, &["EXE", "check"]);
    chk(r#""EXE check""#, &["EXE check"]);
    chk(r#""EXE """for""" check"#, &["EXE for check"]);
    chk(r#""EXE \"for\" check"#, &[r"EXE \for\ check"]);
    chk(r#""EXE \" for \" check"#, &[r"EXE \", "for", r#"""#, "check"]);
    chk(r#"E"X"E test"#, &["EXE", "test"]);
    chk(r#"EX""E test"#, &["EXE", "test"]);
}

// from https://daviddeley.com/autohotkey/parameters/parameters.htm#WINCRULESEX
#[test]
fn post_2008() {
    chk("EXE CallMeIshmael", &["EXE", "CallMeIshmael"]);
    chk(r#"EXE "Call Me Ishmael""#, &["EXE", "Call Me Ishmael"]);
    chk(r#"EXE Cal"l Me I"shmael"#, &["EXE", "Call Me Ishmael"]);
    chk(r#"EXE CallMe\"Ishmael"#, &["EXE", r#"CallMe"Ishmael"#]);
    chk(r#"EXE "CallMe\"Ishmael""#, &["EXE", r#"CallMe"Ishmael"#]);
    chk(r#"EXE "Call Me Ishmael\\""#, &["EXE", r"Call Me Ishmael\"]);
    chk(r#"EXE "CallMe\\\"Ishmael""#, &["EXE", r#"CallMe\"Ishmael"#]);
    chk(r#"EXE a\\\b"#, &["EXE", r"a\\\b"]);
    chk(r#"EXE "a\\\b""#, &["EXE", r"a\\\b"]);
    chk(r#"EXE "\"Call Me Ishmael\"""#, &["EXE", r#""Call Me Ishmael""#]);
    chk(r#"EXE "C:\TEST A\\""#, &["EXE", r"C:\TEST A\"]);
    chk(r#"EXE "\"C:\TEST A\\\"""#, &["EXE", r#""C:\TEST A\""#]);
    chk(r#"EXE "a b c"  d  e"#, &["EXE", "a b c", "d", "e"]);
    chk(r#"EXE "ab\"c"  "\\"  d"#, &["EXE", r#"ab"c"#, r"\", "d"]);
    chk(r#"EXE a\\\b d"e f"g h"#, &["EXE", r"a\\\b", "de fg", "h"]);
    chk(r#"EXE a\\\"b c d"#, &["EXE", r#"a\"b"#, "c", "d"]);
    chk(r#"EXE a\\\\"b c" d e"#, &["EXE", r"a\\b c", "d", "e"]);
    // Double Double Quotes
    chk(r#"EXE "a b c"""#, &["EXE", r#"a b c""#]);
    chk(r#"EXE """CallMeIshmael"""  b  c"#, &["EXE", r#""CallMeIshmael""#, "b", "c"]);
    chk(r#"EXE """Call Me Ishmael""""#, &["EXE", r#""Call Me Ishmael""#]);
    chk(r#"EXE """"Call Me Ishmael"" b c"#, &["EXE", r#""Call"#, "Me", "Ishmael", "b", "c"]);
}
