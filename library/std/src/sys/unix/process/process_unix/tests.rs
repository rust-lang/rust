#[test]
#[rustfmt::skip] // avoids tidy destroying the legibility of the hex/string tables
fn exitstatus_display_tests() {
    // In practice this is the same on every Unix.
    // If some weird platform turns out to be different, and this test fails, use if cfg!
    use crate::os::unix::process::ExitStatusExt;
    use crate::process::ExitStatus;

    let t = |v, exp: &[&str]| {
        let got = format!("{}", <ExitStatus as ExitStatusExt>::from_raw(v));
        assert!(exp.contains(&got.as_str()), "got={:?} exp={:?}", &got, exp);
    };

    // SuS says that wait status 0 corresponds to WIFEXITED and WEXITSTATUS==0.
    // The implementation of `ExitStatusError` relies on this fact.
    // So this one must always pass - don't disable this one with cfg!
    t(0x00000, &["exit status: 0"]);

    // We cope with a variety of conventional signal strings, both with and without the signal
    // abbrevation too.  It would be better to compare this with the result of strsignal but that
    // is not threadsafe which is big reason we want a set of test cases...
    //
    // The signal formatting is done by signal_display in library/std/src/sys/unix/os.rs.
    t(0x0000f, &["signal: Terminated",
                 "signal: Terminated (SIGTERM)"]);
    t(0x0008b, &["signal: Segmentation fault (core dumped)",
                 "signal: Segmentation fault (SIGSEGV) (core dumped)"]);
    t(0x0ff00, &["exit status: 255"]);

    // On MacOS, 0x0137f is WIFCONTINUED, not WIFSTOPPED.  Probably *BSD is similar.
    //   https://github.com/rust-lang/rust/pull/82749#issuecomment-790525956
    // The purpose of this test is to test our string formatting, not our understanding of the wait
    // status magic numbers.  So restrict these to Linux.
    if cfg!(target_os = "linux") {
        t(0x0137f, &["stopped (not terminated) by signal: Stopped (signal)",
                     "stopped (not terminated) by signal: Stopped (signal) (SIGSTOP)"]);
        t(0x0ffff, &["continued (WIFCONTINUED)"]);
    }

    // Testing "unrecognised wait status" is hard because the wait.h macros typically
    // assume that the value came from wait and isn't mad.  With the glibc I have here
    // this works:
    if cfg!(all(target_os = "linux", target_env = "gnu")) {
        t(0x000ff, &["unrecognised wait status: 255 0xff"]);
    }
}
