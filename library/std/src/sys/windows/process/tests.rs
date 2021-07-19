use super::make_command_line;
use super::Arg;
use crate::env;
use crate::ffi::{OsStr, OsString};
use crate::process::Command;

#[test]
fn test_raw_args() {
    let command_line = &make_command_line(
        OsStr::new("quoted exe"),
        &[
            Arg::Regular(OsString::from("quote me")),
            Arg::Raw(OsString::from("quote me *not*")),
            Arg::Raw(OsString::from("\t\\")),
            Arg::Raw(OsString::from("internal \\\"backslash-\"quote")),
            Arg::Regular(OsString::from("optional-quotes")),
        ],
        false,
    )
    .unwrap();
    assert_eq!(
        String::from_utf16(command_line).unwrap(),
        "\"quoted exe\" \"quote me\" quote me *not* \t\\ internal \\\"backslash-\"quote optional-quotes"
    );
}

#[test]
fn test_make_command_line() {
    fn test_wrapper(prog: &str, args: &[&str], force_quotes: bool) -> String {
        let command_line = &make_command_line(
            OsStr::new(prog),
            &args.iter().map(|a| Arg::Regular(OsString::from(a))).collect::<Vec<_>>(),
            force_quotes,
        )
        .unwrap();
        String::from_utf16(command_line).unwrap()
    }

    assert_eq!(test_wrapper("prog", &["aaa", "bbb", "ccc"], false), "\"prog\" aaa bbb ccc");

    assert_eq!(test_wrapper("prog", &[r"C:\"], false), r#""prog" C:\"#);
    assert_eq!(test_wrapper("prog", &[r"2slashes\\"], false), r#""prog" 2slashes\\"#);
    assert_eq!(test_wrapper("prog", &[r" C:\"], false), r#""prog" " C:\\""#);
    assert_eq!(test_wrapper("prog", &[r" 2slashes\\"], false), r#""prog" " 2slashes\\\\""#);

    assert_eq!(
        test_wrapper("C:\\Program Files\\blah\\blah.exe", &["aaa"], false),
        "\"C:\\Program Files\\blah\\blah.exe\" aaa"
    );
    assert_eq!(
        test_wrapper("C:\\Program Files\\blah\\blah.exe", &["aaa", "v*"], false),
        "\"C:\\Program Files\\blah\\blah.exe\" aaa v*"
    );
    assert_eq!(
        test_wrapper("C:\\Program Files\\blah\\blah.exe", &["aaa", "v*"], true),
        "\"C:\\Program Files\\blah\\blah.exe\" \"aaa\" \"v*\""
    );
    assert_eq!(
        test_wrapper("C:\\Program Files\\test", &["aa\"bb"], false),
        "\"C:\\Program Files\\test\" aa\\\"bb"
    );
    assert_eq!(test_wrapper("echo", &["a b c"], false), "\"echo\" \"a b c\"");
    assert_eq!(
        test_wrapper("echo", &["\" \\\" \\", "\\"], false),
        "\"echo\" \"\\\" \\\\\\\" \\\\\" \\"
    );
    assert_eq!(
        test_wrapper("\u{03c0}\u{042f}\u{97f3}\u{00e6}\u{221e}", &[], false),
        "\"\u{03c0}\u{042f}\u{97f3}\u{00e6}\u{221e}\""
    );
}

// On Windows, environment args are case preserving but comparisons are case-insensitive.
// See: #85242
#[test]
fn windows_env_unicode_case() {
    let test_cases = [
        ("ä", "Ä"),
        ("ß", "SS"),
        ("Ä", "Ö"),
        ("Ä", "Ö"),
        ("I", "İ"),
        ("I", "i"),
        ("I", "ı"),
        ("i", "I"),
        ("i", "İ"),
        ("i", "ı"),
        ("İ", "I"),
        ("İ", "i"),
        ("İ", "ı"),
        ("ı", "I"),
        ("ı", "i"),
        ("ı", "İ"),
        ("ä", "Ä"),
        ("ß", "SS"),
        ("Ä", "Ö"),
        ("Ä", "Ö"),
        ("I", "İ"),
        ("I", "i"),
        ("I", "ı"),
        ("i", "I"),
        ("i", "İ"),
        ("i", "ı"),
        ("İ", "I"),
        ("İ", "i"),
        ("İ", "ı"),
        ("ı", "I"),
        ("ı", "i"),
        ("ı", "İ"),
    ];
    // Test that `cmd.env` matches `env::set_var` when setting two strings that
    // may (or may not) be case-folded when compared.
    for (a, b) in test_cases.iter() {
        let mut cmd = Command::new("cmd");
        cmd.env(a, "1");
        cmd.env(b, "2");
        env::set_var(a, "1");
        env::set_var(b, "2");

        for (key, value) in cmd.get_envs() {
            assert_eq!(
                env::var(key).ok(),
                value.map(|s| s.to_string_lossy().into_owned()),
                "command environment mismatch: {} {}",
                a,
                b
            );
        }
    }
}
