//! ThingOS process implementation for `std::process`.
//!
//! # Spawn contract
//!
//! `SYS_SPAWN_PROCESS_EX` (0x1006) launches child processes.
//!
//! ## Request (SpawnProcessExReq, abi/src/types/system.rs)
//!
//!  - name_ptr/name_len : Executable path (UTF-8, not null-terminated)
//!  - argv_ptr/argv_len : count:u32, then (len:u32, bytes)xN
//!  - env_ptr/env_len   : count:u32, then (klen, key, vlen, val)xN
//!  - stdin/stdout/stderr_mode : 0=inherit, 1=null, 2=pipe
//!
//! ## Response (SpawnProcessExResp)
//!
//!  - child_pid   : use with SYS_WAITPID
//!  - stdin_pipe  : parent WRITE fd when mode==pipe (else 0)
//!  - stdout_pipe : parent READ fd when mode==pipe (else 0)
//!  - stderr_pipe : parent READ fd when mode==pipe (else 0)
//!
//! ## cwd override
//!
//! When `Command::current_dir()` is set, the cwd bytes are forwarded via the
//! `cwd_ptr`/`cwd_len` fields of `SpawnProcessExReq`.  The kernel sets the
//! child's working directory atomically at spawn time.  If cwd is not set,
//! the child inherits the parent's cwd.
//!
//! ## From<ChildPipe> for Stdio
//!
//! Passing an existing io::PipeReader/PipeWriter as child stdio is not
//! supported (no per-fd remapping in the ThingOS spawn ABI).  Falls back
//! to Stdio::Inherit.  Use Stdio::piped() for pipe-based capture.

use super::env::{CommandEnv, CommandEnvs};
pub use crate::ffi::OsString as EnvKey;
use crate::ffi::{OsStr, OsString};
use crate::num::NonZero;
use crate::path::Path;
use crate::process::StdioPipes;
use crate::sys::fs::File;
use crate::sys::pal::raw_syscall6;
use crate::{fmt, io};

// Syscall numbers (abi/src/numbers.rs)
const SYS_SPAWN_PROCESS_EX: u32 = 0x1006;
const SYS_WAITPID: u32 = 0x1011;
const SYS_TASK_KILL: u32 = 0x1008;

/// waitpid WNOHANG: return immediately if no child has exited yet.
const WNOHANG: usize = 1;

/// stdio mode constants (abi/src/types/system.rs `stdio_mode`)
mod stdio_mode {
    pub const INHERIT: u32 = 0;
    pub const NULL: u32 = 1;
    pub const PIPE: u32 = 2;
}

#[inline]
fn cvt(ret: isize) -> crate::io::Result<usize> {
    if ret < 0 { Err(crate::io::Error::from_raw_os_error((-ret) as i32)) } else { Ok(ret as usize) }
}

const SYS_FS_POLL: u32 = 0x400B;
const SYS_FS_FCNTL: u32 = 0x4018;

const F_GETFL: u32 = 3;
const F_SETFL: u32 = 4;
const O_NONBLOCK: u32 = 0x0800;

const POLLIN: u16 = 0x0001;
const POLLERR: u16 = 0x0008;
const POLLHUP: u16 = 0x0010;

#[repr(C)]
struct PollFd {
    fd: i32,
    events: u16,
    revents: u16,
}

// ── ABI structs (must match abi/src/types/system.rs) ─────────────────────────

#[repr(C)]
#[derive(Default)]
struct SpawnProcessExReq {
    name_ptr: u64,
    name_len: u32,
    _pad0: u32,
    argv_ptr: u64,
    argv_len: u32,
    _pad1: u32,
    env_ptr: u64,
    env_len: u32,
    _pad2: u32,
    stdin_mode: u32,
    stdout_mode: u32,
    stderr_mode: u32,
    _reserved: u32,
    boot_arg: u64,
    handles_to_inherit: [u64; 8],
    num_inherited_handles: u32,
    _pad3: u32,
    /// Pointer to desired working directory bytes (NOT null-terminated).
    /// Set to 0 to inherit the parent's cwd.
    cwd_ptr: u64,
    /// Length of the cwd bytes (0 = inherit parent cwd).
    cwd_len: u32,
    _pad4: u32,
}

#[repr(C)]
#[derive(Default)]
struct SpawnProcessExResp {
    child_tid: u64,
    child_pid: u32,
    _pad: u32,
    stdin_pipe: u64,
    stdout_pipe: u64,
    stderr_pipe: u64,
}

// ── Serialization helpers ─────────────────────────────────────────────────────

fn serialize_argv(args: &[OsString]) -> crate::vec::Vec<u8> {
    let mut blob = crate::vec::Vec::new();
    blob.extend_from_slice(&(args.len() as u32).to_le_bytes());
    for arg in args {
        let bytes = arg.as_encoded_bytes();
        blob.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        blob.extend_from_slice(bytes);
    }
    blob
}

fn serialize_env(env: &crate::collections::BTreeMap<EnvKey, OsString>) -> crate::vec::Vec<u8> {
    let mut blob = crate::vec::Vec::new();
    blob.extend_from_slice(&(env.len() as u32).to_le_bytes());
    for (k, v) in env {
        let kb = k.as_encoded_bytes();
        let vb = v.as_encoded_bytes();
        blob.extend_from_slice(&(kb.len() as u32).to_le_bytes());
        blob.extend_from_slice(kb);
        blob.extend_from_slice(&(vb.len() as u32).to_le_bytes());
        blob.extend_from_slice(vb);
    }
    blob
}

// ── Stdio ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
    ParentStdout,
    ParentStderr,
    #[allow(dead_code)]
    InheritFile(File),
}

impl Stdio {
    fn mode(&self) -> u32 {
        match self {
            Stdio::Inherit
            | Stdio::ParentStdout
            | Stdio::ParentStderr
            | Stdio::InheritFile(_) => stdio_mode::INHERIT,
            Stdio::Null => stdio_mode::NULL,
            Stdio::MakePipe => stdio_mode::PIPE,
        }
    }
}

impl From<ChildPipe> for Stdio {
    fn from(pipe: ChildPipe) -> Stdio {
        // ThingOS spawn ABI has no per-fd remapping; fall back to inherit.
        drop(pipe);
        Stdio::Inherit
    }
}

impl From<io::Stdout> for Stdio {
    fn from(_: io::Stdout) -> Stdio { Stdio::ParentStdout }
}

impl From<io::Stderr> for Stdio {
    fn from(_: io::Stderr) -> Stdio { Stdio::ParentStderr }
}

impl From<File> for Stdio {
    fn from(file: File) -> Stdio { Stdio::InheritFile(file) }
}

// ── Command ───────────────────────────────────────────────────────────────────

pub struct Command {
    program: OsString,
    args: crate::vec::Vec<OsString>,
    env: CommandEnv,
    cwd: Option<OsString>,
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
}

impl Command {
    pub fn new(program: &OsStr) -> Command {
        Command {
            program: program.to_owned(),
            args: crate::vec![program.to_owned()],
            env: Default::default(),
            cwd: None,
            stdin: None,
            stdout: None,
            stderr: None,
        }
    }

    pub fn arg(&mut self, arg: &OsStr) { self.args.push(arg.to_owned()); }

    pub fn env_mut(&mut self) -> &mut CommandEnv { &mut self.env }

    pub fn cwd(&mut self, dir: &OsStr) {
        self.cwd = Some(dir.to_owned());
    }

    pub fn stdin(&mut self, stdin: Stdio) { self.stdin = Some(stdin); }
    pub fn stdout(&mut self, stdout: Stdio) { self.stdout = Some(stdout); }
    pub fn stderr(&mut self, stderr: Stdio) { self.stderr = Some(stderr); }

    pub fn get_program(&self) -> &OsStr { &self.program }

    pub fn get_args(&self) -> CommandArgs<'_> {
        let mut iter = self.args.iter();
        iter.next();
        CommandArgs { iter }
    }

    pub fn get_envs(&self) -> CommandEnvs<'_> { self.env.iter() }
    pub fn get_env_clear(&self) -> bool { self.env.does_clear() }

    pub fn get_current_dir(&self) -> Option<&Path> {
        self.cwd.as_ref().map(|cs| Path::new(cs))
    }

    pub fn spawn(
        &mut self,
        default: Stdio,
        _needs_stdin: bool,
    ) -> crate::io::Result<(Process, StdioPipes)> {
        let stdin_mode = self.stdin.as_ref().unwrap_or(&default).mode();
        let stdout_mode = self.stdout.as_ref().unwrap_or(&default).mode();
        let stderr_mode = self.stderr.as_ref().unwrap_or(&default).mode();

        let argv_blob = serialize_argv(&self.args);
        // Resolve the full child environment (current env + any overrides).
        let full_env = self.env.capture();
        let env_blob = serialize_env(&full_env);
        let name_bytes = self.program.as_encoded_bytes();

        // Encode optional cwd override.
        let cwd_bytes = self.cwd.as_ref().map(|c| c.as_encoded_bytes());

        let req = SpawnProcessExReq {
            name_ptr: name_bytes.as_ptr() as u64,
            name_len: name_bytes.len() as u32,
            argv_ptr: argv_blob.as_ptr() as u64,
            argv_len: argv_blob.len() as u32,
            env_ptr: if env_blob.is_empty() { 0 } else { env_blob.as_ptr() as u64 },
            env_len: env_blob.len() as u32,
            stdin_mode,
            stdout_mode,
            stderr_mode,
            cwd_ptr: cwd_bytes.map_or(0, |b| b.as_ptr() as u64),
            cwd_len: cwd_bytes.map_or(0, |b| b.len() as u32),
            ..Default::default()
        };

        let mut resp = SpawnProcessExResp::default();
        let ret = unsafe {
            raw_syscall6(
                SYS_SPAWN_PROCESS_EX,
                &req as *const SpawnProcessExReq as usize,
                &mut resp as *mut SpawnProcessExResp as usize,
                0, 0, 0, 0,
            )
        };
        cvt(ret)?;

        // SAFETY: kernel guarantees these fds are valid pipe ends.
        let stdin_pipe = if stdin_mode == stdio_mode::PIPE && resp.stdin_pipe != 0 {
            Some(unsafe { crate::sys::pipe::Pipe::from_raw_fd(resp.stdin_pipe as u32) })
        } else {
            None
        };
        let stdout_pipe = if stdout_mode == stdio_mode::PIPE && resp.stdout_pipe != 0 {
            Some(unsafe { crate::sys::pipe::Pipe::from_raw_fd(resp.stdout_pipe as u32) })
        } else {
            None
        };
        let stderr_pipe = if stderr_mode == stdio_mode::PIPE && resp.stderr_pipe != 0 {
            Some(unsafe { crate::sys::pipe::Pipe::from_raw_fd(resp.stderr_pipe as u32) })
        } else {
            None
        };

        let process = Process { pid: resp.child_pid, tid: resp.child_tid, status: None };
        let pipes = StdioPipes { stdin: stdin_pipe, stdout: stdout_pipe, stderr: stderr_pipe };
        Ok((process, pipes))
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref cwd) = self.cwd {
            write!(f, "cd {cwd:?} && ")?;
        }
        write!(f, "{:?}", self.args[0])?;
        for arg in &self.args[1..] {
            write!(f, " {:?}", arg)?;
        }
        Ok(())
    }
}

// ── CommandArgs ───────────────────────────────────────────────────────────────

pub struct CommandArgs<'a> {
    iter: crate::slice::Iter<'a, OsString>,
}

impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;
    fn next(&mut self) -> Option<&'a OsStr> {
        self.iter.next().map(|os| &**os)
    }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<'a> ExactSizeIterator for CommandArgs<'a> {
    fn len(&self) -> usize { self.iter.len() }
    fn is_empty(&self) -> bool { self.iter.is_empty() }
}

impl<'a> fmt::Debug for CommandArgs<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter.clone()).finish()
    }
}

// ── ExitStatus ────────────────────────────────────────────────────────────────

/// Child process exit status.  ThingOS stores the raw exit(code) as an i32.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(i32);

impl ExitStatus {
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        if self.0 == 0 {
            Ok(())
        } else {
            Err(ExitStatusError(NonZero::new(self.0).expect("non-zero exit code")))
        }
    }
    pub fn code(&self) -> Option<i32> { Some(self.0) }
}

impl Default for ExitStatus {
    fn default() -> Self {
        ExitStatus(0)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exit status: {}", self.0)
    }
}

// ── ExitStatusError ───────────────────────────────────────────────────────────

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatusError(NonZero<i32>);

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus { ExitStatus(self.0.get()) }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZero<i32>> { Some(self.0) }
}

// ── ExitCode ──────────────────────────────────────────────────────────────────

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitCode(u8);

impl ExitCode {
    pub const SUCCESS: ExitCode = ExitCode(0);
    pub const FAILURE: ExitCode = ExitCode(1);
    pub fn as_i32(&self) -> i32 { self.0 as i32 }
}

impl From<u8> for ExitCode {
    fn from(code: u8) -> Self { Self(code) }
}

// ── Process (child handle) ────────────────────────────────────────────────────

pub struct Process {
    pid: u32,
    /// Main thread TID of the child, used for kill().
    tid: u64,
    /// Cached exit status after the first successful wait().
    status: Option<ExitStatus>,
}

impl Process {
    pub fn id(&self) -> u32 { self.pid }

    /// Request termination of the child process via SYS_TASK_KILL (uses TID).
    pub fn kill(&mut self) -> crate::io::Result<()> {
        let ret = unsafe { raw_syscall6(SYS_TASK_KILL, self.tid as usize, 0, 0, 0, 0, 0) };
        cvt(ret).map(|_| ())
    }

    /// Block until child exits; returns cached status on second call.
    pub fn wait(&mut self) -> crate::io::Result<ExitStatus> {
        if let Some(status) = self.status { return Ok(status); }
        let mut code: i32 = 0;
        let ret = unsafe {
            raw_syscall6(SYS_WAITPID, self.pid as usize,
                &mut code as *mut i32 as usize, 0, 0, 0, 0)
        };
        cvt(ret)?;
        let status = ExitStatus(code);
        self.status = Some(status);
        Ok(status)
    }

    /// Non-blocking check; returns Ok(None) if child is still running.
    pub fn try_wait(&mut self) -> crate::io::Result<Option<ExitStatus>> {
        if let Some(status) = self.status { return Ok(Some(status)); }
        let mut code: i32 = 0;
        let ret = unsafe {
            raw_syscall6(SYS_WAITPID, self.pid as usize,
                &mut code as *mut i32 as usize, WNOHANG, 0, 0, 0)
        };
        let child_pid = cvt(ret)?;
        if child_pid == 0 {
            Ok(None)
        } else {
            let status = ExitStatus(code);
            self.status = Some(status);
            Ok(Some(status))
        }
    }
}

// ── ChildPipe + read_output + output ─────────────────────────────────────────

/// A pipe end held by the parent for child stdio I/O.
pub type ChildPipe = crate::sys::pipe::Pipe;

/// Drain stdout and stderr concurrently (multiplexed drain).
///
/// This prevents deadlocks where the child blocks writing to stderr
/// because the parent is blocked reading from stdout, or vice versa.
pub fn read_output(
    out: ChildPipe,
    stdout: &mut crate::vec::Vec<u8>,
    err: ChildPipe,
    stderr: &mut crate::vec::Vec<u8>,
) -> crate::io::Result<()> {
    let out_fd = out.0 as i32;
    let err_fd = err.0 as i32;

    // Set non-blocking mode on both pipes so we can drain them concurrently
    // without one blocking the other's progress.
    for fd in &[out_fd, err_fd] {
        let fl = unsafe { raw_syscall6(SYS_FS_FCNTL, *fd as usize, F_GETFL as usize, 0, 0, 0, 0) };
        if fl < 0 {
            return Err(crate::io::Error::from_raw_os_error(-fl as i32));
        }
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_FCNTL,
                *fd as usize,
                F_SETFL as usize,
                (fl as usize) | O_NONBLOCK as usize,
                0,
                0,
                0,
            )
        };
        if ret < 0 {
            return Err(crate::io::Error::from_raw_os_error(-ret as i32));
        }
    }

    let mut out_done = false;
    let mut err_done = false;
    let mut tmp = [0u8; 4096];

    while !out_done || !err_done {
        let mut pfds = [
            PollFd { fd: out_fd, events: if out_done { 0 } else { POLLIN }, revents: 0 },
            PollFd { fd: err_fd, events: if err_done { 0 } else { POLLIN }, revents: 0 },
        ];

        // Wait for data or hangup on either pipe.
        let ret = unsafe {
            raw_syscall6(SYS_FS_POLL, pfds.as_mut_ptr() as usize, 2, usize::MAX, 0, 0, 0)
        };
        if ret < 0 {
            let err = -ret as i32;
            if err == 4 {
                continue;
            } // EINTR
            return Err(crate::io::Error::from_raw_os_error(err));
        }

        // Drain stdout if there's activity.
        if !out_done && (pfds[0].revents & (POLLIN | POLLERR | POLLHUP) != 0) {
            loop {
                match out.read(&mut tmp) {
                    Ok(0) => {
                        out_done = true;
                        break;
                    }
                    Ok(n) => {
                        stdout.extend_from_slice(&tmp[..n]);
                    }
                    Err(e) if e.raw_os_error() == Some(11) => break, // EAGAIN
                    Err(e) => return Err(e),
                }
            }
        }

        // Drain stderr if there's activity.
        if !err_done && (pfds[1].revents & (POLLIN | POLLERR | POLLHUP) != 0) {
            loop {
                match err.read(&mut tmp) {
                    Ok(0) => {
                        err_done = true;
                        break;
                    }
                    Ok(n) => {
                        stderr.extend_from_slice(&tmp[..n]);
                    }
                    Err(e) if e.raw_os_error() == Some(11) => break, // EAGAIN
                    Err(e) => return Err(e),
                }
            }
        }
    }

    Ok(())
}

/// Implementation of Command::output() on ThingOS.
pub fn output(
    cmd: &mut Command,
) -> crate::io::Result<(ExitStatus, crate::vec::Vec<u8>, crate::vec::Vec<u8>)> {
    let (mut process, mut pipes) = cmd.spawn(Stdio::MakePipe, false)?;
    drop(pipes.stdin.take());
    let mut stdout = crate::vec::Vec::new();
    let mut stderr = crate::vec::Vec::new();
    if let (Some(out), Some(err)) = (pipes.stdout.take(), pipes.stderr.take()) {
        read_output(out, &mut stdout, err, &mut stderr)?;
    }
    let status = process.wait()?;
    Ok((status, stdout, stderr))
}
