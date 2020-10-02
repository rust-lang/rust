#![unstable(feature = "process_internals", issue = "none")]

#[cfg(test)]
mod tests;

use crate::borrow::Borrow;
use crate::collections::BTreeMap;
use crate::env;
use crate::env::split_paths;
use crate::ffi::{OsStr, OsString};
use crate::fmt;
use crate::fs;
use crate::io::{self, Error, ErrorKind};
use crate::mem;
use crate::os::windows::ffi::OsStrExt;
use crate::path::Path;
use crate::ptr;
use crate::sys::c;
use crate::sys::cvt;
use crate::sys::fs::{File, OpenOptions};
use crate::sys::handle::Handle;
use crate::sys::mutex::Mutex;
use crate::sys::pipe::{self, AnonPipe};
use crate::sys::stdio;
use crate::sys_common::process::{CommandEnv, CommandEnvs};
use crate::sys_common::AsInner;
use core::convert::TryInto;

use libc::{c_void, EXIT_FAILURE, EXIT_SUCCESS};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[doc(hidden)]
pub struct EnvKey(OsString);

impl From<OsString> for EnvKey {
    fn from(mut k: OsString) -> Self {
        k.make_ascii_uppercase();
        EnvKey(k)
    }
}

impl From<EnvKey> for OsString {
    fn from(k: EnvKey) -> Self {
        k.0
    }
}

impl Borrow<OsStr> for EnvKey {
    fn borrow(&self) -> &OsStr {
        &self.0
    }
}

impl AsRef<OsStr> for EnvKey {
    fn as_ref(&self) -> &OsStr {
        &self.0
    }
}

fn ensure_no_nuls<T: AsRef<OsStr>>(str: T) -> Result<T, Problem> {
    if str.as_ref().encode_wide().any(|b| b == 0) { Err(Problem::SawNul) } else { Ok(str) }
}

// 32768 minus NUL plus starting space in our implementation
const CMDLINE_MAX: usize = 32768;
pub struct Command {
    program: OsString,
    args: Vec<OsString>,
    env: CommandEnv,
    cwd: Option<OsString>,
    flags: u32,
    detach: bool, // not currently exposed in std::process
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
    cmdline: Vec<u16>,
    problem: Option<Problem>,
}

pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
    Handle(Handle),
}

pub struct StdioPipes {
    pub stdin: Option<AnonPipe>,
    pub stdout: Option<AnonPipe>,
    pub stderr: Option<AnonPipe>,
}

struct DropGuard<'a> {
    lock: &'a Mutex,
}

pub enum Problem {
    SawNul,
    Oversized,
}

/// Types that can be appended to a Windows command-line. Used for custom escaping.
// FIXME: the force-quoted one should probably be its own type.
#[unstable(feature = "windows_raw_cmdline", issue = "74549")]
pub trait Arg {
    fn append_to(&self, cmd: &mut Vec<u16>, force_quotes: bool) -> Result<usize, Problem>;
    #[unstable(feature = "command_sized", issue = "74549")]
    fn arg_size(&self, force_quotes: bool) -> Result<usize, Problem>;
    fn to_os_string(&self) -> OsString;
}

/// Argument type with no escaping.
#[unstable(feature = "windows_raw_cmdline", issue = "74549")]
pub struct RawArg<'a>(&'a OsStr);

impl Command {
    pub fn new(program: &OsStr) -> Command {
        Command {
            program: program.to_os_string(),
            args: Vec::new(),
            env: Default::default(),
            cwd: None,
            flags: 0,
            detach: false,
            stdin: None,
            stdout: None,
            stderr: None,
            cmdline: Vec::new(),
            problem: None,
        }
    }

    #[allow(dead_code)]
    pub fn maybe_arg_ext(&mut self, arg: impl Arg) -> io::Result<()> {
        self.arg_ext(arg);

        match &self.problem {
            Some(err) => Err(err.into()),
            None => Ok(()),
        }
    }
    pub fn arg_ext(&mut self, arg: impl Arg) {
        if self.problem.is_some() {
            return;
        }

        self.args.push(arg.to_os_string());
        self.cmdline.push(' ' as u16);
        let result = arg.append_to(&mut self.cmdline, false);
        match result {
            Err(err) => {
                self.cmdline.truncate(self.cmdline.len() - 1);
                self.problem = Some(err);
            }
            Ok(length) => {
                if self.cmdline.len() >= CMDLINE_MAX {
                    // Roll back oversized
                    self.cmdline.truncate(self.cmdline.len() - 1 - length);
                    self.problem = Some(Problem::Oversized)
                }
            }
        };
    }
    #[allow(dead_code)]
    pub fn maybe_arg(&mut self, arg: &OsStr) -> io::Result<()> {
        self.maybe_arg_ext(arg)
    }
    pub fn arg(&mut self, arg: &OsStr) {
        self.arg_ext(arg)
    }
    pub fn env_mut(&mut self) -> &mut CommandEnv {
        &mut self.env
    }
    pub fn cwd(&mut self, dir: &OsStr) {
        self.cwd = Some(dir.to_os_string())
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
    pub fn creation_flags(&mut self, flags: u32) {
        self.flags = flags;
    }
    #[allow(dead_code)]
    pub fn problem(&self) -> io::Result<()> {
        if let Some(err) = &self.problem {
            return Err(err.into());
        }
        Ok(())
    }

    pub fn get_program(&self) -> &OsStr {
        &self.program
    }

    pub fn get_args(&self) -> CommandArgs<'_> {
        let iter = self.args.iter();
        CommandArgs { iter }
    }

    pub fn get_envs(&self) -> CommandEnvs<'_> {
        self.env.iter()
    }

    pub fn get_current_dir(&self) -> Option<&Path> {
        self.cwd.as_ref().map(|cwd| Path::new(cwd))
    }

    pub fn spawn(
        &mut self,
        default: Stdio,
        needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        if let Some(err) = &self.problem {
            return Err(err.into());
        }

        let maybe_env = self.env.capture_if_changed();
        // To have the spawning semantics of unix/windows stay the same, we need
        // to read the *child's* PATH if one is provided. See #15149 for more
        // details.
        let rprogram = maybe_env.as_ref().and_then(|env| {
            if let Some(v) = env.get(OsStr::new("PATH")) {
                // Split the value and test each path to see if the
                // program exists.
                for path in split_paths(&v) {
                    let path = path
                        .join(self.program.to_str().unwrap())
                        .with_extension(env::consts::EXE_EXTENSION);
                    if fs::metadata(&path).is_ok() {
                        return Some(path.into_os_string());
                    }
                }
            }
            None
        });
        let program = rprogram.as_ref().unwrap_or(&self.program);

        // Prepare and terminate the application name and the cmdline
        // FIXME: this won't work for 16-bit, which requires the program
        // to be put on the cmdline. Do an extend_from_slice?
        let mut program_str: Vec<u16> = Vec::new();
        program.as_os_str().append_to(&mut program_str, true)?;
        program_str.push(0);
        self.cmdline.push(0);

        let mut si = zeroed_startupinfo();
        si.cb = mem::size_of::<c::STARTUPINFO>() as c::DWORD;
        si.dwFlags = c::STARTF_USESTDHANDLES;

        // stolen from the libuv code.
        let mut flags = self.flags | c::CREATE_UNICODE_ENVIRONMENT;
        if self.detach {
            flags |= c::DETACHED_PROCESS | c::CREATE_NEW_PROCESS_GROUP;
        }

        let (envp, _data) = make_envp(maybe_env)?;
        let (dirp, _data) = make_dirp(self.cwd.as_ref())?;
        let mut pi = zeroed_process_information();

        // Prepare all stdio handles to be inherited by the child. This
        // currently involves duplicating any existing ones with the ability to
        // be inherited by child processes. Note, however, that once an
        // inheritable handle is created, *any* spawned child will inherit that
        // handle. We only want our own child to inherit this handle, so we wrap
        // the remaining portion of this spawn in a mutex.
        //
        // For more information, msdn also has an article about this race:
        // http://support.microsoft.com/kb/315939
        static CREATE_PROCESS_LOCK: Mutex = Mutex::new();
        let _guard = DropGuard::new(&CREATE_PROCESS_LOCK);

        let mut pipes = StdioPipes { stdin: None, stdout: None, stderr: None };
        let null = Stdio::Null;
        let default_stdin = if needs_stdin { &default } else { &null };
        let stdin = self.stdin.as_ref().unwrap_or(default_stdin);
        let stdout = self.stdout.as_ref().unwrap_or(&default);
        let stderr = self.stderr.as_ref().unwrap_or(&default);
        let stdin = stdin.to_handle(c::STD_INPUT_HANDLE, &mut pipes.stdin)?;
        let stdout = stdout.to_handle(c::STD_OUTPUT_HANDLE, &mut pipes.stdout)?;
        let stderr = stderr.to_handle(c::STD_ERROR_HANDLE, &mut pipes.stderr)?;
        si.hStdInput = stdin.raw();
        si.hStdOutput = stdout.raw();
        si.hStdError = stderr.raw();

        unsafe {
            cvt(c::CreateProcessW(
                program_str.as_mut_ptr(),
                self.cmdline.as_mut_ptr().offset(1), // Skip the starting space
                ptr::null_mut(),
                ptr::null_mut(),
                c::TRUE,
                flags,
                envp,
                dirp,
                &mut si,
                &mut pi,
            ))
        }?;

        // We close the thread handle because we don't care about keeping
        // the thread id valid, and we aren't keeping the thread handle
        // around to be able to close it later.
        drop(Handle::new(pi.hThread));

        Ok((Process { handle: Handle::new(pi.hProcess) }, pipes))
    }

    pub fn get_size(&mut self) -> io::Result<usize> {
        match &self.problem {
            Some(err) => Err(err.into()),
            None => Ok(self.cmdline.len()),
        }
    }
    pub fn available_size(&mut self, _refresh: bool) -> io::Result<isize> {
        let size: isize = match self.get_size()?.try_into() {
            Ok(s) => Ok(s),
            Err(_) => Err(io::Error::from(Problem::Oversized)),
        }?;

        Ok((CMDLINE_MAX as isize) - size)
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.program)?;
        for arg in &self.args {
            write!(f, " {:?}", arg)?;
        }
        Ok(())
    }
}

impl<'a> DropGuard<'a> {
    fn new(lock: &'a Mutex) -> DropGuard<'a> {
        unsafe {
            lock.lock();
            DropGuard { lock }
        }
    }
}

impl<'a> Drop for DropGuard<'a> {
    fn drop(&mut self) {
        unsafe {
            self.lock.unlock();
        }
    }
}

impl Stdio {
    fn to_handle(&self, stdio_id: c::DWORD, pipe: &mut Option<AnonPipe>) -> io::Result<Handle> {
        match *self {
            // If no stdio handle is available, then inherit means that it
            // should still be unavailable so propagate the
            // INVALID_HANDLE_VALUE.
            Stdio::Inherit => match stdio::get_handle(stdio_id) {
                Ok(io) => {
                    let io = Handle::new(io);
                    let ret = io.duplicate(0, true, c::DUPLICATE_SAME_ACCESS);
                    io.into_raw();
                    ret
                }
                Err(..) => Ok(Handle::new(c::INVALID_HANDLE_VALUE)),
            },

            Stdio::MakePipe => {
                let ours_readable = stdio_id != c::STD_INPUT_HANDLE;
                let pipes = pipe::anon_pipe(ours_readable, true)?;
                *pipe = Some(pipes.ours);
                Ok(pipes.theirs.into_handle())
            }

            Stdio::Handle(ref handle) => handle.duplicate(0, true, c::DUPLICATE_SAME_ACCESS),

            // Open up a reference to NUL with appropriate read/write
            // permissions as well as the ability to be inherited to child
            // processes (as this is about to be inherited).
            Stdio::Null => {
                let size = mem::size_of::<c::SECURITY_ATTRIBUTES>();
                let mut sa = c::SECURITY_ATTRIBUTES {
                    nLength: size as c::DWORD,
                    lpSecurityDescriptor: ptr::null_mut(),
                    bInheritHandle: 1,
                };
                let mut opts = OpenOptions::new();
                opts.read(stdio_id == c::STD_INPUT_HANDLE);
                opts.write(stdio_id != c::STD_INPUT_HANDLE);
                opts.security_attributes(&mut sa);
                File::open(Path::new("NUL"), &opts).map(|file| file.into_handle())
            }
        }
    }
}

impl From<AnonPipe> for Stdio {
    fn from(pipe: AnonPipe) -> Stdio {
        Stdio::Handle(pipe.into_handle())
    }
}

impl From<File> for Stdio {
    fn from(file: File) -> Stdio {
        Stdio::Handle(file.into_handle())
    }
}

impl From<&Problem> for Error {
    fn from(problem: &Problem) -> Error {
        match *problem {
            Problem::SawNul => {
                Error::new(ErrorKind::InvalidInput, "nul byte found in provided data")
            }
            Problem::Oversized => {
                Error::new(ErrorKind::InvalidInput, "command exceeds maximum size")
            }
        }
    }
}

impl From<Problem> for Error {
    fn from(problem: Problem) -> Error {
        (&problem).into()
    }
}

impl Arg for &OsStr {
    fn append_to(&self, cmd: &mut Vec<u16>, force_quotes: bool) -> Result<usize, Problem> {
        append_arg(&mut Some(cmd), &self, force_quotes)
    }
    fn arg_size(&self, force_quotes: bool) -> Result<usize, Problem> {
        Ok(append_arg(&mut None, &self, force_quotes)? + 1)
    }
    fn to_os_string(&self) -> OsString {
        OsStr::to_os_string(&self)
    }
}

#[allow(dead_code)]
impl Arg for RawArg<'_> {
    fn append_to(&self, cmd: &mut Vec<u16>, _fq: bool) -> Result<usize, Problem> {
        cmd.extend(self.0.encode_wide());
        self.arg_size(_fq)
    }
    fn arg_size(&self, _: bool) -> Result<usize, Problem> {
        Ok(self.0.encode_wide().count() + 1)
    }
    fn to_os_string(&self) -> OsString {
        OsStr::to_os_string(&(self.0))
    }
}

impl<'a, T> Arg for &'a T
where
    T: Arg,
{
    fn append_to(&self, cmd: &mut Vec<u16>, _fq: bool) -> Result<usize, Problem> {
        (*self).append_to(cmd, _fq)
    }
    fn arg_size(&self, _fq: bool) -> Result<usize, Problem> {
        (*self).arg_size(_fq)
    }
    fn to_os_string(&self) -> OsString {
        (*self).to_os_string()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

/// A value representing a child process.
///
/// The lifetime of this value is linked to the lifetime of the actual
/// process - the Process destructor calls self.finish() which waits
/// for the process to terminate.
pub struct Process {
    handle: Handle,
}

impl Process {
    pub fn kill(&mut self) -> io::Result<()> {
        cvt(unsafe { c::TerminateProcess(self.handle.raw(), 1) })?;
        Ok(())
    }

    pub fn id(&self) -> u32 {
        unsafe { c::GetProcessId(self.handle.raw()) as u32 }
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        unsafe {
            let res = c::WaitForSingleObject(self.handle.raw(), c::INFINITE);
            if res != c::WAIT_OBJECT_0 {
                return Err(Error::last_os_error());
            }
            let mut status = 0;
            cvt(c::GetExitCodeProcess(self.handle.raw(), &mut status))?;
            Ok(ExitStatus(status))
        }
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        unsafe {
            match c::WaitForSingleObject(self.handle.raw(), 0) {
                c::WAIT_OBJECT_0 => {}
                c::WAIT_TIMEOUT => {
                    return Ok(None);
                }
                _ => return Err(Error::last_os_error()),
            }
            let mut status = 0;
            cvt(c::GetExitCodeProcess(self.handle.raw(), &mut status))?;
            Ok(Some(ExitStatus(status)))
        }
    }

    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    pub fn into_handle(self) -> Handle {
        self.handle
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(c::DWORD);

impl ExitStatus {
    pub fn success(&self) -> bool {
        self.0 == 0
    }
    pub fn code(&self) -> Option<i32> {
        Some(self.0 as i32)
    }
}

/// Converts a raw `c::DWORD` to a type-safe `ExitStatus` by wrapping it without copying.
impl From<c::DWORD> for ExitStatus {
    fn from(u: c::DWORD) -> ExitStatus {
        ExitStatus(u)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Windows exit codes with the high bit set typically mean some form of
        // unhandled exception or warning. In this scenario printing the exit
        // code in decimal doesn't always make sense because it's a very large
        // and somewhat gibberish number. The hex code is a bit more
        // recognizable and easier to search for, so print that.
        if self.0 & 0x80000000 != 0 {
            write!(f, "exit code: {:#x}", self.0)
        } else {
            write!(f, "exit code: {}", self.0)
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitCode(c::DWORD);

impl ExitCode {
    pub const SUCCESS: ExitCode = ExitCode(EXIT_SUCCESS as _);
    pub const FAILURE: ExitCode = ExitCode(EXIT_FAILURE as _);

    #[inline]
    pub fn as_i32(&self) -> i32 {
        self.0 as i32
    }
}

fn zeroed_startupinfo() -> c::STARTUPINFO {
    c::STARTUPINFO {
        cb: 0,
        lpReserved: ptr::null_mut(),
        lpDesktop: ptr::null_mut(),
        lpTitle: ptr::null_mut(),
        dwX: 0,
        dwY: 0,
        dwXSize: 0,
        dwYSize: 0,
        dwXCountChars: 0,
        dwYCountCharts: 0,
        dwFillAttribute: 0,
        dwFlags: 0,
        wShowWindow: 0,
        cbReserved2: 0,
        lpReserved2: ptr::null_mut(),
        hStdInput: c::INVALID_HANDLE_VALUE,
        hStdOutput: c::INVALID_HANDLE_VALUE,
        hStdError: c::INVALID_HANDLE_VALUE,
    }
}

fn zeroed_process_information() -> c::PROCESS_INFORMATION {
    c::PROCESS_INFORMATION {
        hProcess: ptr::null_mut(),
        hThread: ptr::null_mut(),
        dwProcessId: 0,
        dwThreadId: 0,
    }
}

macro_rules! if_some {
    ($e: expr, $id:ident, $b:block) => {
        if let &mut Some(ref mut $id) = $e
            $b
    };
    ($e: expr, $id:ident, $s:stmt) => {
        if_some!($e, $id, { $s })
    };
}

// This is effed up. Yeah, how the heck do I pass an optional, mutable reference around?
// @see https://users.rust-lang.org/t/idiomatic-way-for-passing-an-optional-mutable-reference-around/7947
fn append_arg(
    maybe_cmd: &mut Option<&mut Vec<u16>>,
    arg: &OsStr,
    force_quotes: bool,
) -> Result<usize, Problem> {
    let mut addsize: usize = 0;
    // If an argument has 0 characters then we need to quote it to ensure
    // that it actually gets passed through on the command line or otherwise
    // it will be dropped entirely when parsed on the other end.
    ensure_no_nuls(arg)?;
    let arg_bytes = &arg.as_inner().inner.as_inner();
    let quote =
        force_quotes || arg_bytes.iter().any(|c| *c == b' ' || *c == b'\t') || arg_bytes.is_empty();
    if quote {
        if_some!(maybe_cmd, cmd, cmd.push('"' as u16));
        addsize += 1;
    }

    let mut backslashes: usize = 0;
    for x in arg.encode_wide() {
        if x == '\\' as u16 {
            backslashes += 1;
        } else {
            if x == '"' as u16 {
                // Add n+1 backslashes to total 2n+1 before internal '"'.
                if_some!(maybe_cmd, cmd, cmd.extend((0..=backslashes).map(|_| '\\' as u16)));
                addsize += backslashes + 1;
            }
            backslashes = 0;
        }
        if_some!(maybe_cmd, cmd, cmd.push(x));
    }

    if quote {
        // Add n backslashes to total 2n before ending '"'.
        if_some!(maybe_cmd, cmd, {
            cmd.extend((0..backslashes).map(|_| '\\' as u16));
            cmd.push('"' as u16);
        });
        addsize += backslashes + 1;
    }
    Ok(addsize)
}

// Produces a wide string *without terminating null*; returns an error if
// `prog` or any of the `args` contain a nul.
#[allow(dead_code)]
fn make_command_line(prog: &OsStr, args: &[OsString]) -> io::Result<Vec<u16>> {
    // Encode the command and arguments in a command line string such
    // that the spawned process may recover them using CommandLineToArgvW.
    let mut cmd: Vec<u16> = Vec::new();
    // Always quote the program name so CreateProcess doesn't interpret args as
    // part of the name if the binary wasn't found first time.
    prog.append_to(&mut cmd, true)?;
    for arg in args {
        cmd.push(' ' as u16);
        arg.as_os_str().append_to(&mut cmd, false)?;
    }
    return Ok(cmd);
}

fn make_envp(maybe_env: Option<BTreeMap<EnvKey, OsString>>) -> io::Result<(*mut c_void, Vec<u16>)> {
    // On Windows we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    if let Some(env) = maybe_env {
        let mut blk = Vec::new();

        for (k, v) in env {
            blk.extend(ensure_no_nuls(k.0)?.encode_wide());
            blk.push('=' as u16);
            blk.extend(ensure_no_nuls(v)?.encode_wide());
            blk.push(0);
        }
        blk.push(0);
        Ok((blk.as_mut_ptr() as *mut c_void, blk))
    } else {
        Ok((ptr::null_mut(), Vec::new()))
    }
}

fn make_dirp(d: Option<&OsString>) -> io::Result<(*const u16, Vec<u16>)> {
    match d {
        Some(dir) => {
            let mut dir_str: Vec<u16> = ensure_no_nuls(dir)?.encode_wide().collect();
            dir_str.push(0);
            Ok((dir_str.as_ptr(), dir_str))
        }
        None => Ok((ptr::null(), Vec::new())),
    }
}

pub struct CommandArgs<'a> {
    iter: crate::slice::Iter<'a, OsString>,
}

impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;
    fn next(&mut self) -> Option<&'a OsStr> {
        self.iter.next().map(|s| s.as_ref())
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
