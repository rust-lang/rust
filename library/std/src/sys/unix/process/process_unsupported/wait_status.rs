//! Emulated wait status for non-Unix #[cfg(unix) platforms
//!
//! Separate module to facilitate testing against a real Unix implementation.

use crate::ffi::c_int;
use crate::fmt;

/// Emulated wait status for use by `process_unsupported.rs`
///
/// Uses the "traditional unix" encoding.  For use on platfors which are `#[cfg(unix)]`
/// but do not actually support subprocesses at all.
///
/// These platforms aren't Unix, but are simply pretending to be for porting convenience.
/// So, we provide a faithful pretence here.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub struct ExitStatus {
    wait_status: c_int,
}

/// Converts a raw `c_int` to a type-safe `ExitStatus` by wrapping it
impl From<c_int> for ExitStatus {
    fn from(wait_status: c_int) -> ExitStatus {
        ExitStatus { wait_status }
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "emulated wait status: {}", self.wait_status)
    }
}

impl ExitStatus {
    pub fn code(&self) -> Option<i32> {
        // Linux and FreeBSD both agree that values linux 0x80
        // count as "WIFEXITED" even though this is quite mad.
        // Likewise the macros disregard all the high bits, so are happy to declare
        // out-of-range values to be WIFEXITED, WIFSTOPPED, etc.
        let w = self.wait_status;
        if (w & 0x7f) == 0 { Some((w & 0xff00) >> 8) } else { None }
    }

    pub fn signal(&self) -> Option<i32> {
        let signal = self.wait_status & 0x007f;
        if signal > 0 && signal < 0x7f { Some(signal) } else { None }
    }

    pub fn core_dumped(&self) -> bool {
        self.signal().is_some() && (self.wait_status & 0x80) != 0
    }

    pub fn stopped_signal(&self) -> Option<i32> {
        let w = self.wait_status;
        if (w & 0xff) == 0x7f { Some((w & 0xff00) >> 8) } else { None }
    }

    pub fn continued(&self) -> bool {
        self.wait_status == 0xffff
    }

    pub fn into_raw(&self) -> c_int {
        self.wait_status
    }
}

// Test that our emulation exactly matches Linux
//
// This test runs *on Linux* but it tests
// the implementation used on non-Unix `#[cfg(unix)]` platforms.
//
// I.e. we're using Linux as a proxy for "trad unix".
#[cfg(all(test, target_os = "linux"))]
mod compare_with_linux {
    use super::ExitStatus as Emulated;
    use crate::os::unix::process::ExitStatusExt as _;
    use crate::process::ExitStatus as Real;

    #[test]
    fn compare() {
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
}
