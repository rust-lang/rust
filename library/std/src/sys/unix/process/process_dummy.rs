use crate::fmt;
use crate::io;
use crate::sys::process::process_common::*;
use crate::os::raw::c_int;

pub use crate::ffi::OsString as EnvKey;

fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

fn unsupported_err() -> io::Error {
    io::Error::new(io::ErrorKind::Other, "operation not supported")
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Void {}

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

/// Unix exit statuses
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(c_int);

impl ExitStatus {
    fn exited(&self) -> bool {
        libc::WIFEXITED(self.0)
    }

    pub fn success(&self) -> bool {
        self.code() == Some(0)
    }

    pub fn code(&self) -> Option<i32> {
        if self.exited() { Some(libc::WEXITSTATUS(self.0)) } else { None }
    }

    pub fn signal(&self) -> Option<i32> {
        if !self.exited() { Some(libc::WTERMSIG(self.0)) } else { None }
    }
}

impl From<c_int> for ExitStatus {
    fn from(a: c_int) -> ExitStatus {
        ExitStatus(a)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(code) = self.code() {
            write!(f, "exit code: {}", code)
        } else {
            let signal = self.signal().unwrap();
            write!(f, "signal: {}", signal)
        }
    }
}

pub struct Process(Void);

impl Process {
    pub fn id(&self) -> u32 {
        match self.0 {}
    }

    pub fn kill(&mut self) -> io::Result<()> {
        match self.0 {}
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        match self.0 {}
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        match self.0 {}
    }
}
