#![allow(missing_docs)]

use crate::sys_common::process::{CommandEnv, CommandEnvs};
use crate::ffi::{OsStr, OsString};
use crate::sys::os::error_string;
use crate::sys::pipe::AnonPipe;
use crate::num::NonZeroI32;
use crate::custom_os_impl;
use crate::sys::fs::File;
use crate::path::Path;
use crate::fmt;
use crate::io;

pub type EnvKey = OsString;

/// Inner content of [`crate::process::Command`]
#[derive(Debug)]
pub struct Command {
    /// Environment (variables)
    pub env: CommandEnv,
    /// Name/Path of the program to run
    pub program: OsString,
    /// Initial working directory
    pub cwd: Option<OsString>,
    /// Arguments
    pub args: Vec<OsString>,
    /// Will have a defined value when passed to the custom platform
    pub stdin: Option<Stdio>,
    /// Will have a defined value when passed to the custom platform
    pub stdout: Option<Stdio>,
    /// Will have a defined value when passed to the custom platform
    pub stderr: Option<Stdio>,
    pub needs_stdin: bool,
}

/// Passed back to std::process with the pipes connected to
/// the child, if any were requested
pub struct StdioPipes {
    pub stdin: Option<AnonPipe>,
    pub stdout: Option<AnonPipe>,
    pub stderr: Option<AnonPipe>,
}

/// Defines the source for IO pipes of the child process
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
}

impl Command {
    pub(crate) fn new(program: &OsStr) -> Command {
        Command {
            env: Default::default(),
            program: program.into(),
            cwd: None,
            args: Vec::new(),
            stdin: None,
            stdout: None,
            stderr: None,
            needs_stdin: /* will be overwritten */ false,
        }
    }

    pub(crate) fn arg(&mut self, arg: &OsStr) {
        self.args.push(arg.into())
    }

    pub(crate) fn env_mut(&mut self) -> &mut CommandEnv {
        &mut self.env
    }

    pub(crate) fn cwd(&mut self, dir: &OsStr) {
        self.cwd = Some(dir.into())
    }

    pub(crate) fn stdin(&mut self, stdin: Stdio) {
        self.stdin = Some(stdin)
    }

    pub(crate) fn stdout(&mut self, stdout: Stdio) {
        self.stdout = Some(stdout)
    }

    pub(crate) fn stderr(&mut self, stderr: Stdio) {
        self.stderr = Some(stderr)
    }

    pub(crate) fn get_program(&self) -> &OsStr {
        &self.program
    }

    pub(crate) fn get_args(&self) -> CommandArgs<'_> {
        CommandArgs { iter: self.args.iter() }
    }

    pub(crate) fn get_envs(&self) -> CommandEnvs<'_> {
        self.env.iter()
    }

    pub(crate) fn get_current_dir(&self) -> Option<&Path> {
        self.cwd.as_ref().map(|string| Path::new(string))
    }

    pub(crate) fn spawn(
        &mut self,
        default: Stdio,
        needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        self.stdin .get_or_insert(default);
        self.stdout.get_or_insert(default);
        self.stderr.get_or_insert(default);
        self.needs_stdin = needs_stdin;

        custom_os_impl!(process, spawn, self)
    }

    pub(crate) fn output(&mut self) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
        todo!()
    }
}

impl From<AnonPipe> for Stdio {
    fn from(pipe: AnonPipe) -> Stdio {
        pipe.diverge()
    }
}

impl From<File> for Stdio {
    fn from(_file: File) -> Stdio {
        todo!()
    }
}

/// Success must be internally represented as zero
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
#[repr(transparent)]
#[non_exhaustive]
pub struct ExitStatus(pub i32);

impl ExitStatus {
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        match self.0 == ExitCode::SUCCESS.as_i32() {
            true => Ok(()),
            false => Err(ExitStatusError(*self)),
        }
    }

    pub fn code(&self) -> Option<i32> {
        Some(self.0)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", error_string(self.0))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ExitStatusError(ExitStatus);

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        self.0
    }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZeroI32> {
        self.0.code().map(|c| {
            NonZeroI32::try_from(c).expect("invalid ExitStatus")
        })
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

impl From<u8> for ExitCode {
    fn from(code: u8) -> Self {
        match code {
            0 => Self::SUCCESS,
            1..=255 => Self::FAILURE,
        }
    }
}

/// Inner content of [`crate::process::Child`]
pub type Process = Box<dyn ProcessApi>;

/// Object-oriented manipulation of a [`Process`]
pub trait ProcessApi {
    fn id(&self) -> u32;
    fn kill(&mut self) -> io::Result<()>;
    fn wait(&mut self) -> io::Result<ExitStatus>;
    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>>;
}

pub struct CommandArgs<'a> {
    iter: crate::slice::Iter<'a, OsString>,
}

impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;
    fn next(&mut self) -> Option<&'a OsStr> {
        self.iter.next().map(|string| &**string)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> ExactSizeIterator for CommandArgs<'a> {
    fn len(&self) -> usize {
        self.iter.len()
    }
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

impl<'a> fmt::Debug for CommandArgs<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter.clone()).finish()
    }
}
