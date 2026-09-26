use super::*;
use crate::ffi::OsString;

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

fn bat_cmd(script: &str, args: &[&str], force_quotes: bool) -> crate::io::Result<String> {
    // build a bat command line and decode it back to a UTF-8 String for easy assertions.
    let script_wide: Vec<u16> = OsString::from(script).encode_wide().collect();
    let args: Vec<Arg> = args.iter().map(|s| Arg::Regular(OsString::from(s))).collect();
    let cmd = make_bat_command_line(&script_wide, &args, force_quotes)?;
    Ok(String::from_utf16_lossy(&cmd).into_owned())
}

#[test]
fn bat_no_args() {
    let result = bat_cmd("script.bat", &[], false).unwrap();
    assert_eq!(result, r#"cmd.exe /e:ON /v:OFF /d /c ""script.bat""#);
}

#[test]
fn bat_simple_arg() {
    let result = bat_cmd("script.bat", &["hello"], false).unwrap();
    assert_eq!(result, r#"cmd.exe /e:ON /v:OFF /d /c ""script.bat" hello""#);
}

#[test]
fn bat_arg_with_spaces() {
    let result = bat_cmd("script.bat", &["hello world"], false).unwrap();
    assert_eq!(result, r#"cmd.exe /e:ON /v:OFF /d /c ""script.bat" "hello world"""#);
}

#[test]
fn bat_empty_arg() {
    let result = bat_cmd("script.bat", &[""], false).unwrap();
    assert_eq!(result, r#"cmd.exe /e:ON /v:OFF /d /c ""script.bat" """#);
}

#[test]
fn bat_arg_with_double_quote() {
    let result = bat_cmd("script.bat", &[r#"say "hi""#], false).unwrap();
    assert_eq!(result, r#"cmd.exe /e:ON /v:OFF /d /c ""script.bat" "say ""hi""""#);
}

#[test]
fn bat_arg_with_backslash_before_quote() {
    let result = bat_cmd("script.bat", &[r#"a\"#], false).unwrap();
    assert_eq!(result, r#"cmd.exe /e:ON /v:OFF /d /c ""script.bat" "a\\"""#);
}

#[test]
fn bat_arg_with_percent() {
    let result = bat_cmd("script.bat", &["%PATH%"], false).unwrap();
    assert!(result.contains("%%cd:~,"), "percent should be escaped: {result}");
    assert!(!result.contains("%PATH%"), "raw %PATH% must not appear: {result}");
}

#[test]
fn bat_force_quotes() {
    let result = bat_cmd("script.bat", &["plain"], true).unwrap();
    assert_eq!(result, r#"cmd.exe /e:ON /v:OFF /d /c ""script.bat" "plain"""#);
}

#[test]
fn bat_multiple_args() {
    let result = bat_cmd("script.bat", &["one", "two", "three"], false).unwrap();
    assert_eq!(result, r#"cmd.exe /e:ON /v:OFF /d /c ""script.bat" one two three""#);
}

#[test]
fn bat_rejects_newline_in_arg() {
    assert!(bat_cmd("script.bat", &["bad\narg"], false).is_err());
    assert!(bat_cmd("script.bat", &["bad\rarg"], false).is_err());
}

#[test]
fn bat_rejects_script_with_quote() {
    let script_wide: Vec<u16> = OsString::from(r#"scr"ipt.bat"#).encode_wide().collect();
    assert!(make_bat_command_line(&script_wide, &[], false).is_err());
}

#[test]
fn bat_rejects_script_ending_with_backslash() {
    let script_wide: Vec<u16> = OsString::from(r"dir\").encode_wide().collect();
    assert!(make_bat_command_line(&script_wide, &[], false).is_err());
}
