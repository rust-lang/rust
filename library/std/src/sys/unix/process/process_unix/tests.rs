#[test]
fn exitstatus_display_tests() {
    // In practice this is the same on every Unix.
    // If some weird platform turns out to be different, and this test fails, use #[cfg].
    use crate::os::unix::process::ExitStatusExt;
    use crate::process::ExitStatus;

    let t = |v, s| assert_eq!(s, format!("{}", <ExitStatus as ExitStatusExt>::from_raw(v)));

    t(0x0000f, "signal: 15");
    t(0x0008b, "signal: 11 (core dumped)");
    t(0x00000, "exit code: 0");
    t(0x0ff00, "exit code: 255");

    // On MacOS, 0x0137f is WIFCONTINUED, not WIFSTOPPED.  Probably *BSD is similar.
    //   https://github.com/rust-lang/rust/pull/82749#issuecomment-790525956
    // The purpose of this test is to test our string formatting, not our understanding of the wait
    // status magic numbers.  So restrict these to Linux.
    if cfg!(target_os = "linux") {
        t(0x0137f, "stopped (not terminated) by signal: 19");
        t(0x0ffff, "continued (WIFCONTINUED)");
    }

    // Testing "unrecognised wait status" is hard because the wait.h macros typically
    // assume that the value came from wait and isn't mad.  With the glibc I have here
    // this works:
    if cfg!(all(target_os = "linux", target_env = "gnu")) {
        t(0x000ff, "unrecognised wait status: 255 0xff");
    }
}
