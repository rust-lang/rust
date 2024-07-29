//! Emulated wait status for non-Unix #[cfg(unix) platforms
//!
//! Separate module to facilitate testing against a real Unix implementation.

use super::ExitStatusError;
use crate::ffi::c_int;
use crate::fmt;
use crate::num::NonZero;

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

    #[allow(unused)]
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        // This assumes that WIFEXITED(status) && WEXITSTATUS==0 corresponds to status==0. This is
        // true on all actual versions of Unix, is widely assumed, and is specified in SuS
        // https://pubs.opengroup.org/onlinepubs/9699919799/functions/wait.html. If it is not
        // true for a platform pretending to be Unix, the tests (our doctests, and also
        // process_unix/tests.rs) will spot it. `ExitStatusError::code` assumes this too.
        match NonZero::try_from(self.wait_status) {
            /* was nonzero */ Ok(failure) => Err(ExitStatusError(failure)),
            /* was zero, couldn't convert */ Err(_) => Ok(()),
        }
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

#[cfg(test)]
#[path = "wait_status/tests.rs"] // needed because of strange layout of process_unsupported
mod tests;
