// Note that tests in this file are run on Linux as well as on platforms using unsupported

// Test that our emulation exactly matches Linux
//
// This test runs *on Linux* but it tests
// the implementation used on non-Unix `#[cfg(unix)]` platforms.
//
// I.e. we're using Linux as a proxy for "trad unix".
#[cfg(target_os = "linux")]
#[test]
fn compare_with_linux() {
    use super::ExitStatus as Emulated;
    use crate::os::unix::process::ExitStatusExt as _;
    use crate::process::ExitStatus as Real;

    // Check that we handle out-of-range values similarly, too.
    for wstatus in -0xf_ffff..0xf_ffff {
        let emulated = Emulated::from(wstatus);
        let real = Real::from_raw(wstatus);

        macro_rules! compare { { $method:ident } => {
            assert_eq!(
                emulated.$method(),
                real.$method(),
                "{wstatus:#x}.{}()",
                stringify!($method),
            );
        } }
        compare!(code);
        compare!(signal);
        compare!(core_dumped);
        compare!(stopped_signal);
        compare!(continued);
        compare!(into_raw);
    }
}
