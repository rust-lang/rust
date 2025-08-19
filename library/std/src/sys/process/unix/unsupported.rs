use libc::{c_int, pid_t};

use super::common::*;
use crate::io;
use crate::num::NonZero;
use crate::sys::pal::unsupported::*;

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

impl Command {
    pub fn spawn(
        &mut self,
        _default: Stdio,
        _needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        unsupported()
    }

    pub fn exec(&mut self, _default: Stdio) -> io::Error {
        unsupported_err()
    }
}

pub fn output(_: &mut Command) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
    unsupported()
}

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

pub struct Process {
    _handle: pid_t,
}

impl Process {
    pub fn id(&self) -> u32 {
        0
    }

    pub fn kill(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn send_signal(&self, _signal: i32) -> io::Result<()> {
        unsupported()
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        unsupported()
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        unsupported()
    }
}

mod wait_status;
pub use wait_status::ExitStatus;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatusError(NonZero<c_int>);

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        ExitStatus::from(c_int::from(self.0))
    }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZero<i32>> {
        ExitStatus::from(c_int::from(self.0)).code().map(|st| st.try_into().unwrap())
    }
}
