use super::{Arg, make_command_line};
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
fn test_thread_handle() {
    use crate::os::windows::io::BorrowedHandle;
    use crate::os::windows::process::{ChildExt, CommandExt};
    const CREATE_SUSPENDED: u32 = 0x00000004;

    let p = Command::new("cmd").args(&["/C", "exit 0"]).creation_flags(CREATE_SUSPENDED).spawn();
    assert!(p.is_ok());
    let mut p = p.unwrap();

    extern "system" {
        fn ResumeThread(_: BorrowedHandle<'_>) -> u32;
    }
    unsafe {
        ResumeThread(p.main_thread_handle());
    }

    crate::thread::sleep(crate::time::Duration::from_millis(100));

    let res = p.try_wait();
    assert!(res.is_ok());
    assert!(res.unwrap().is_some());
    assert!(p.try_wait().unwrap().unwrap().success());
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
                "command environment mismatch: {a} {b}",
            );
        }
    }
}

// UWP applications run in a restricted environment which means this test may not work.
#[cfg(not(target_vendor = "uwp"))]
#[test]
fn windows_exe_resolver() {
    use super::resolve_exe;
    use crate::io;
    use crate::sys::fs::symlink;
    use crate::sys_common::io::test::tmpdir;

    let env_paths = || env::var_os("PATH");

    // Test a full path, with and without the `exe` extension.
    let mut current_exe = env::current_exe().unwrap();
    assert!(resolve_exe(current_exe.as_ref(), env_paths, None).is_ok());
    current_exe.set_extension("");
    assert!(resolve_exe(current_exe.as_ref(), env_paths, None).is_ok());

    // Test lone file names.
    assert!(resolve_exe(OsStr::new("cmd"), env_paths, None).is_ok());
    assert!(resolve_exe(OsStr::new("cmd.exe"), env_paths, None).is_ok());
    assert!(resolve_exe(OsStr::new("cmd.EXE"), env_paths, None).is_ok());
    assert!(resolve_exe(OsStr::new("fc"), env_paths, None).is_ok());

    // Invalid file names should return InvalidInput.
    assert_eq!(
        resolve_exe(OsStr::new(""), env_paths, None).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        resolve_exe(OsStr::new("\0"), env_paths, None).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    // Trailing slash, therefore there's no file name component.
    assert_eq!(
        resolve_exe(OsStr::new(r"C:\Path\to\"), env_paths, None).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );

    /*
    Some of the following tests may need to be changed if you are deliberately
    changing the behavior of `resolve_exe`.
    */

    let empty_paths = || None;

    // The resolver looks in system directories even when `PATH` is empty.
    assert!(resolve_exe(OsStr::new("cmd.exe"), empty_paths, None).is_ok());

    // The application's directory is also searched.
    let current_exe = env::current_exe().unwrap();
    assert!(resolve_exe(current_exe.file_name().unwrap().as_ref(), empty_paths, None).is_ok());

    // Create a temporary path and add a broken symlink.
    let temp = tmpdir();
    let mut exe_path = temp.path().to_owned();
    exe_path.push("exists.exe");

    // A broken symlink should still be resolved.
    // Skip this check if not in CI and creating symlinks isn't possible.
    let is_ci = env::var("CI").is_ok();
    let result = symlink("<DOES NOT EXIST>".as_ref(), &exe_path);
    if is_ci || result.is_ok() {
        result.unwrap();
        assert!(
            resolve_exe(OsStr::new("exists.exe"), empty_paths, Some(temp.path().as_ref())).is_ok()
        );
    }
}
