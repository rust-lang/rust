use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use crate::command::{Command, CompletedProcess};
use crate::{cwd, env_var};

#[track_caller]
fn run_common(name: &str, args: Option<&[&str]>) -> Command {
    let mut bin_path = PathBuf::new();
    bin_path.push(cwd());
    bin_path.push(name);
    let ld_lib_path_envvar = env_var("LD_LIB_PATH_ENVVAR");

    let mut cmd = if let Some(rtc) = env::var_os("REMOTE_TEST_CLIENT") {
        let mut cmd = Command::new(rtc);
        cmd.arg("run");
        // FIXME: the "0" indicates how many support files should be uploaded along with the binary
        // to execute. If a test requires additional files to be pushed to the remote machine, this
        // will have to be changed (and the support files will have to be uploaded).
        cmd.arg("0");
        cmd.arg(bin_path);
        cmd
    } else if let Ok(runner) = std::env::var("RUNNER") {
        let mut args = split_maybe_args(&runner);

        let prog = args.remove(0);
        let mut cmd = Command::new(prog);

        for arg in args {
            cmd.arg(arg);
        }

        cmd.arg("--");
        cmd.arg(bin_path);

        cmd
    } else {
        Command::new(bin_path)
    };

    if let Some(args) = args {
        for arg in args {
            cmd.arg(arg);
        }
    }

    cmd.env(&ld_lib_path_envvar, {
        let mut paths = vec![];
        paths.push(cwd());
        for p in env::split_paths(&env_var("TARGET_EXE_DYLIB_PATH")) {
            paths.push(p.to_path_buf());
        }
        for p in env::split_paths(&env_var(&ld_lib_path_envvar)) {
            paths.push(p.to_path_buf());
        }
        env::join_paths(paths.iter()).unwrap()
    });
    cmd.env("LC_ALL", "C"); // force english locale

    cmd.inspect(|std_cmd| eprintln!("running: {std_cmd:?}"));
    cmd
}

/// Run a built binary and make sure it succeeds.
#[track_caller]
pub fn run(name: &str) -> CompletedProcess {
    run_common(name, None).run()
}

/// Run a built binary with one or more argument(s) and make sure it succeeds.
#[track_caller]
pub fn run_with_args(name: &str, args: &[&str]) -> CompletedProcess {
    run_common(name, Some(args)).run()
}

/// Run a built binary and make sure it fails.
#[track_caller]
pub fn run_fail(name: &str) -> CompletedProcess {
    run_common(name, None).run_fail()
}

/// Create a new custom [`Command`]. This should be preferred to creating [`std::process::Command`]
/// directly.
#[track_caller]
pub fn cmd<S: AsRef<OsStr>>(program: S) -> Command {
    let mut command = Command::new(program);
    command.env("LC_ALL", "C"); // force english locale
    command
}

fn split_maybe_args(s: &str) -> Vec<OsString> {
    // Split on default shell IFS (space, tab, newline).
    s.split(|c: char| matches!(c, ' ' | '\t' | '\n'))
        .filter(|s| !s.is_empty())
        .map(OsString::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(s: &str) -> OsString {
        OsString::from(s)
    }

    // --- Tests that PASS on the current (buggy) code ---

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

    // --- Tests that FAIL on the current (buggy) code ---
    // They prove the bug: split_maybe_args only splits on space,
    // but shells split on IFS which defaults to space, tab, and newline.

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
}
