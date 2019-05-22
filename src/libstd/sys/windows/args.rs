#![allow(dead_code)] // runtime init functions not used during testing

use crate::os::windows::prelude::*;
use crate::sys::windows::os::current_exe;
use crate::sys::c;
use crate::ffi::OsString;
use crate::fmt;
use crate::vec;
use crate::slice;
use crate::path::PathBuf;

use core::iter;

pub unsafe fn init(_argc: isize, _argv: *const *const u8) { }

pub unsafe fn cleanup() { }

pub fn args() -> Args {
    unsafe {
        let lp_cmd_line = c::GetCommandLineW();
        let parsed_args_list = parse_lp_cmd_line(
            lp_cmd_line as *const u16,
            || current_exe().map(PathBuf::into_os_string).unwrap_or_else(|_| OsString::new()));

        Args { parsed_args_list: parsed_args_list.into_iter() }
    }
}

/// Implements the Windows command-line argument parsing algorithm.
///
/// Microsoft's documentation for the Windows CLI argument format can be found at
/// <https://docs.microsoft.com/en-us/previous-versions//17w5ykft(v=vs.85)>.
///
/// Windows includes a function to do this in shell32.dll,
/// but linking with that DLL causes the process to be registered as a GUI application.
/// GUI applications add a bunch of overhead, even if no windows are drawn. See
/// <https://randomascii.wordpress.com/2018/12/03/a-not-called-function-can-cause-a-5x-slowdown/>.
///
/// This function was tested for equivalence to the shell32.dll implementation in
/// Windows 10 Pro v1803, using an exhaustive test suite available at
/// <https://gist.github.com/notriddle/dde431930c392e428055b2dc22e638f5> or
/// <https://paste.gg/p/anonymous/47d6ed5f5bd549168b1c69c799825223>.
unsafe fn parse_lp_cmd_line<F: Fn() -> OsString>(lp_cmd_line: *const u16, exe_name: F)
                                                 -> Vec<OsString> {
    const BACKSLASH: u16 = '\\' as u16;
    const QUOTE: u16 = '"' as u16;
    const TAB: u16 = '\t' as u16;
    const SPACE: u16 = ' ' as u16;
    let mut ret_val = Vec::new();
    if lp_cmd_line.is_null() || *lp_cmd_line == 0 {
        ret_val.push(exe_name());
        return ret_val;
    }
    let mut cmd_line = {
        let mut end = 0;
        while *lp_cmd_line.offset(end) != 0 {
            end += 1;
        }
        slice::from_raw_parts(lp_cmd_line, end as usize)
    };
    // The executable name at the beginning is special.
    cmd_line = match cmd_line[0] {
        // The executable name ends at the next quote mark,
        // no matter what.
        QUOTE => {
            let args = {
                let mut cut = cmd_line[1..].splitn(2, |&c| c == QUOTE);
                if let Some(exe) = cut.next() {
                    ret_val.push(OsString::from_wide(exe));
                }
                cut.next()
            };
            if let Some(args) = args {
                args
            } else {
                return ret_val;
            }
        }
        // Implement quirk: when they say whitespace here,
        // they include the entire ASCII control plane:
        // "However, if lpCmdLine starts with any amount of whitespace, CommandLineToArgvW
        // will consider the first argument to be an empty string. Excess whitespace at the
        // end of lpCmdLine is ignored."
        0..=SPACE => {
            ret_val.push(OsString::new());
            &cmd_line[1..]
        },
        // The executable name ends at the next whitespace,
        // no matter what.
        _ => {
            let args = {
                let mut cut = cmd_line.splitn(2, |&c| c > 0 && c <= SPACE);
                if let Some(exe) = cut.next() {
                    ret_val.push(OsString::from_wide(exe));
                }
                cut.next()
            };
            if let Some(args) = args {
                args
            } else {
                return ret_val;
            }
        }
    };
    let mut cur = Vec::new();
    let mut in_quotes = false;
    let mut was_in_quotes = false;
    let mut backslash_count: usize = 0;
    for &c in cmd_line {
        match c {
            // backslash
            BACKSLASH => {
                backslash_count += 1;
                was_in_quotes = false;
            },
            QUOTE if backslash_count % 2 == 0 => {
                cur.extend(iter::repeat(b'\\' as u16).take(backslash_count / 2));
                backslash_count = 0;
                if was_in_quotes {
                    cur.push('"' as u16);
                    was_in_quotes = false;
                } else {
                    was_in_quotes = in_quotes;
                    in_quotes = !in_quotes;
                }
            }
            QUOTE if backslash_count % 2 != 0 => {
                cur.extend(iter::repeat(b'\\' as u16).take(backslash_count / 2));
                backslash_count = 0;
                was_in_quotes = false;
                cur.push(b'"' as u16);
            }
            SPACE | TAB if !in_quotes => {
                cur.extend(iter::repeat(b'\\' as u16).take(backslash_count));
                if !cur.is_empty() || was_in_quotes {
                    ret_val.push(OsString::from_wide(&cur[..]));
                    cur.truncate(0);
                }
                backslash_count = 0;
                was_in_quotes = false;
            }
            _ => {
                cur.extend(iter::repeat(b'\\' as u16).take(backslash_count));
                backslash_count = 0;
                was_in_quotes = false;
                cur.push(c);
            }
        }
    }
    cur.extend(iter::repeat(b'\\' as u16).take(backslash_count));
    // include empty quoted strings at the end of the arguments list
    if !cur.is_empty() || was_in_quotes || in_quotes {
        ret_val.push(OsString::from_wide(&cur[..]));
    }
    ret_val
}

pub struct Args {
    parsed_args_list: vec::IntoIter<OsString>,
}

pub struct ArgsInnerDebug<'a> {
    args: &'a Args,
}

impl<'a> fmt::Debug for ArgsInnerDebug<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.args.parsed_args_list.as_slice().fmt(f)
    }
}

impl Args {
    pub fn inner_debug(&self) -> ArgsInnerDebug<'_> {
        ArgsInnerDebug {
            args: self
        }
    }
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> { self.parsed_args_list.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.parsed_args_list.size_hint() }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> { self.parsed_args_list.next_back() }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize { self.parsed_args_list.len() }
}

#[cfg(test)]
mod tests {
    use crate::sys::windows::args::*;
    use crate::ffi::OsString;

    fn chk(string: &str, parts: &[&str]) {
        let mut wide: Vec<u16> = OsString::from(string).encode_wide().collect();
        wide.push(0);
        let parsed = unsafe {
            parse_lp_cmd_line(wide.as_ptr() as *const u16, || OsString::from("TEST.EXE"))
        };
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
            &["EXE", "this is \"all\" in the same argument"]
        );
        chk(r#"EXE "a"""#, &["EXE", "a\""]);
        chk(r#"EXE "a"" a"#, &["EXE", "a\"", "a"]);
        // quotes cannot be escaped in command names
        chk(r#""EXE" check"#, &["EXE", "check"]);
        chk(r#""EXE check""#, &["EXE check"]);
        chk(r#""EXE """for""" check"#, &["EXE ", r#"for""#, "check"]);
        chk(r#""EXE \"for\" check"#, &[r#"EXE \"#, r#"for""#,  "check"]);
    }
}
