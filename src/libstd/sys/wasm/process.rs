use crate::ffi::OsStr;
use crate::fmt;
use crate::io;
use crate::sys::fs::File;
use crate::sys::pipe::AnonPipe;
use crate::sys::{unsupported, Void};
use crate::sys_common::process::{CommandEnv, DefaultEnvKey};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

pub struct Command {
    env: CommandEnv<DefaultEnvKey>
}

// passed back to std::process with the pipes connected to the child, if any
// were requested
pub struct StdioPipes {
    pub stdin: Option<AnonPipe>,
    pub stdout: Option<AnonPipe>,
    pub stderr: Option<AnonPipe>,
}

pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
}

impl Command {
    pub fn new(_program: &OsStr) -> Command {
        Command {
            env: Default::default()
        }
    }

    pub fn arg(&mut self, _arg: &OsStr) {
    }

    pub fn env_mut(&mut self) -> &mut CommandEnv<DefaultEnvKey> {
        &mut self.env
    }

    pub fn cwd(&mut self, _dir: &OsStr) {
    }

    pub fn stdin(&mut self, _stdin: Stdio) {
    }

    pub fn stdout(&mut self, _stdout: Stdio) {
    }

    pub fn stderr(&mut self, _stderr: Stdio) {
    }

    pub fn spawn(&mut self, _default: Stdio, _needs_stdin: bool)
        -> io::Result<(Process, StdioPipes)> {
        unsupported()
    }
}

impl From<AnonPipe> for Stdio {
    fn from(pipe: AnonPipe) -> Stdio {
        pipe.diverge()
    }
}

impl From<File> for Stdio {
    fn from(file: File) -> Stdio {
        file.diverge()
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

pub struct ExitStatus(Void);

impl ExitStatus {
    pub fn success(&self) -> bool {
        match self.0 {}
    }

    pub fn code(&self) -> Option<i32> {
        match self.0 {}
    }
}

impl Clone for ExitStatus {
    fn clone(&self) -> ExitStatus {
        match self.0 {}
    }
}

impl Copy for ExitStatus {}

impl PartialEq for ExitStatus {
    fn eq(&self, _other: &ExitStatus) -> bool {
        match self.0 {}
    }
}

impl Eq for ExitStatus {
}

impl fmt::Debug for ExitStatus {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {}
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {}
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitCode(bool);

impl ExitCode {
    pub const SUCCESS: ExitCode = ExitCode(false);
    pub const FAILURE: ExitCode = ExitCode(true);

    pub fn as_i32(&self) -> i32 {
        self.0 as i32
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
