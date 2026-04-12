//! ThingOS process management.
//!
//! Implements `Command` (spawn/wait/kill) using the `SYS_SPAWN`, `SYS_WAIT`,
//! and `SYS_KILL` system calls.

use super::env::CommandEnv;
pub use super::env::CommandEnvs;
use crate::ffi::{OsStr, OsString};
use crate::path::Path;
use crate::process::StdioPipes;
use crate::sys::fd::FileDesc;
use crate::sys::pal::common::{
    SYS_KILL, SYS_SPAWN, SYS_WAIT, SPAWN_STDIO_INHERIT, SPAWN_STDIO_NULL, SPAWN_STDIO_PIPE, cvt,
    raw_syscall6,
};
use crate::sys::{AsInner, FromInner, IntoInner};
use crate::{fmt, io};
use crate::num::NonZero;

pub use crate::ffi::OsString as EnvKey;

// ── Stdio ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
    Fd(FileDesc),
}

impl Stdio {
    fn as_raw(&self) -> i32 {
        match self {
            Stdio::Inherit => SPAWN_STDIO_INHERIT,
            Stdio::Null => SPAWN_STDIO_NULL,
            Stdio::MakePipe => SPAWN_STDIO_PIPE,
            Stdio::Fd(fd) => fd.as_inner().as_raw_fd(),
        }
    }

    fn try_clone(&self) -> io::Result<Self> {
        match self {
            Self::Fd(fd) => Ok(Self::Fd(FileDesc::from_inner(fd.as_inner().try_clone()?))),
            Self::Inherit => Ok(Self::Inherit),
            Self::Null => Ok(Self::Null),
            Self::MakePipe => Ok(Self::MakePipe),
        }
    }
}

use crate::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

// ── Command ──────────────────────────────────────────────────────────────────

/// Spawn argument block passed to `SYS_SPAWN`.
#[repr(C)]
struct SpawnArgs {
    program_ptr: u64,
    program_len: u64,
    args_ptr: u64,   // pointer to array of (ptr, len) pairs
    args_len: u64,   // number of args
    stdin: i32,
    stdout: i32,
    stderr: i32,
    _pad: i32,
    cwd_ptr: u64,
    cwd_len: u64,
}

#[derive(Default)]
pub struct Command {
    program: OsString,
    args: Vec<OsString>,
    cwd: Option<OsString>,
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
    env: CommandEnv,
}

impl Command {
    pub fn new(program: &OsStr) -> Command {
        Command {
            program: program.to_owned(),
            args: vec![program.to_owned()],
            ..Default::default()
        }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(arg.to_owned());
    }

    pub fn env_mut(&mut self) -> &mut CommandEnv {
        &mut self.env
    }

    pub fn cwd(&mut self, dir: &OsStr) {
        self.cwd = Some(dir.to_owned());
    }

    pub fn stdin(&mut self, stdin: Stdio) {
        self.stdin = Some(stdin);
    }

    pub fn stdout(&mut self, stdout: Stdio) {
        self.stdout = Some(stdout);
    }

    pub fn stderr(&mut self, stderr: Stdio) {
        self.stderr = Some(stderr);
    }

    pub fn get_program(&self) -> &OsStr {
        &self.program
    }

    pub fn get_args(&self) -> CommandArgs<'_> {
        let mut iter = self.args.iter();
        iter.next(); // skip argv[0]
        CommandArgs { iter }
    }

    pub fn get_envs(&self) -> CommandEnvs<'_> {
        self.env.iter()
    }

    pub fn get_env_clear(&self) -> bool {
        self.env.does_clear()
    }

    pub fn get_current_dir(&self) -> Option<&Path> {
        self.cwd.as_ref().map(|cs| Path::new(cs))
    }

    pub fn spawn(
        &mut self,
        default: Stdio,
        _needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        let prog_bytes = self.program.as_os_str().as_encoded_bytes();

        // Build a flat (ptr, len) array for arguments.
        let arg_pairs: Vec<(u64, u64)> = self
            .args
            .iter()
            .map(|a| {
                let b = a.as_os_str().as_encoded_bytes();
                (b.as_ptr() as u64, b.len() as u64)
            })
            .collect();

        let stdin_raw = self.stdin.as_ref().map_or(default.as_raw(), |s| s.as_raw());
        let stdout_raw = self.stdout.as_ref().map_or(default.as_raw(), |s| s.as_raw());
        let stderr_raw = self.stderr.as_ref().map_or(default.as_raw(), |s| s.as_raw());

        let cwd_bytes = self.cwd.as_ref().map(|c| c.as_os_str().as_encoded_bytes());
        let (cwd_ptr, cwd_len) = cwd_bytes
            .map_or((0u64, 0u64), |b| (b.as_ptr() as u64, b.len() as u64));

        let spawn_args = SpawnArgs {
            program_ptr: prog_bytes.as_ptr() as u64,
            program_len: prog_bytes.len() as u64,
            args_ptr: arg_pairs.as_ptr() as u64,
            args_len: arg_pairs.len() as u64,
            stdin: stdin_raw,
            stdout: stdout_raw,
            stderr: stderr_raw,
            _pad: 0,
            cwd_ptr,
            cwd_len,
        };

        let handle = cvt(unsafe {
            raw_syscall6(SYS_SPAWN, &spawn_args as *const SpawnArgs as u64, 0, 0, 0, 0, 0)
        })?;

        // When MakePipe is requested the kernel returns fds alongside the
        // process handle via out-of-band convention; simplified here:
        let pipes = StdioPipes {
            stdin: None,
            stdout: None,
            stderr: None,
        };

        Ok((Process { handle: handle as u64 }, pipes))
    }
}

pub fn output(cmd: &mut Command) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
    use crate::sys::pipe::pipe;
    let (out_reader, out_writer) = pipe()?;
    let (err_reader, err_writer) = pipe()?;
    cmd.stdout(Stdio::Fd(out_writer));
    cmd.stderr(Stdio::Fd(err_writer));

    let (mut process, _pipes) = cmd.spawn(Stdio::Null, false)?;

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();

    use crate::io::Read;
    let _ = out_reader.read_to_end(&mut stdout);
    let _ = err_reader.read_to_end(&mut stderr);

    let status = process.wait()?;
    Ok((status, stdout, stderr))
}

impl From<ChildPipe> for Stdio {
    fn from(pipe: ChildPipe) -> Stdio {
        Stdio::Fd(pipe)
    }
}

impl From<crate::io::Stdout> for Stdio {
    fn from(_: crate::io::Stdout) -> Stdio {
        Stdio::Inherit
    }
}

impl From<crate::io::Stderr> for Stdio {
    fn from(_: crate::io::Stderr) -> Stdio {
        Stdio::Inherit
    }
}

impl From<crate::sys::fs::File> for Stdio {
    fn from(f: crate::sys::fs::File) -> Stdio {
        Stdio::Fd(f.into_inner())
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref cwd) = self.cwd {
            write!(f, "cd {cwd:?} && ")?;
        }
        write!(f, "{:?}", self.program)?;
        for arg in &self.args[1..] {
            write!(f, " {arg:?}")?;
        }
        Ok(())
    }
}

// ── ExitStatus ───────────────────────────────────────────────────────────────

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(i32);

impl ExitStatus {
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        if self.0 == 0 { Ok(()) } else { Err(ExitStatusError(self.0)) }
    }

    pub fn code(&self) -> Option<i32> {
        Some(self.0)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exit code: {}", self.0)
    }
}

// ── ExitStatusError ──────────────────────────────────────────────────────────

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatusError(i32);

impl ExitStatusError {
    pub fn code(self) -> Option<NonZero<i32>> {
        NonZero::new(self.0)
    }
}

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        ExitStatus(self.0)
    }
}

// ── ExitCode ─────────────────────────────────────────────────────────────────

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitCode(u8);

impl ExitCode {
    pub const SUCCESS: ExitCode = ExitCode(0);
    pub const FAILURE: ExitCode = ExitCode(1);

    pub fn as_i32(&self) -> i32 {
        self.0 as i32
    }
}

impl From<u8> for ExitCode {
    fn from(code: u8) -> Self {
        Self(code)
    }
}

// ── Process ──────────────────────────────────────────────────────────────────

pub struct Process {
    handle: u64,
}

impl Process {
    pub fn id(&self) -> u32 {
        self.handle as u32
    }

    pub fn kill(&mut self) -> io::Result<()> {
        cvt(unsafe { raw_syscall6(SYS_KILL, self.handle, 9, 0, 0, 0, 0) })?;
        Ok(())
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        let mut code: i32 = 0;
        cvt(unsafe {
            raw_syscall6(SYS_WAIT, self.handle, &raw mut code as u64, 0, 0, 0, 0)
        })?;
        Ok(ExitStatus(code))
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        // Non-blocking wait: SYS_WAIT with flag 1 = WNOHANG
        let mut code: i32 = -1;
        let ret = unsafe {
            raw_syscall6(SYS_WAIT, self.handle, &raw mut code as u64, 1, 0, 0, 0)
        };
        if ret == 0 {
            Ok(None)
        } else {
            cvt(ret)?;
            Ok(Some(ExitStatus(code)))
        }
    }
}

// ── CommandArgs ──────────────────────────────────────────────────────────────

pub struct CommandArgs<'a> {
    iter: crate::slice::Iter<'a, OsString>,
}

impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;
    fn next(&mut self) -> Option<&'a OsStr> {
        self.iter.next().map(|os| &**os)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> ExactSizeIterator for CommandArgs<'a> {}

impl<'a> fmt::Debug for CommandArgs<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter.clone()).finish()
    }
}

// ── ChildPipe ────────────────────────────────────────────────────────────────

pub type ChildPipe = FileDesc;

pub fn read_output(
    out: ChildPipe,
    stdout: &mut Vec<u8>,
    err: ChildPipe,
    stderr: &mut Vec<u8>,
) -> io::Result<()> {
    use crate::io::Read;
    out.read_to_end(stdout)?;
    err.read_to_end(stderr)?;
    Ok(())
}
