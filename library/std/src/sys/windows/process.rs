#![unstable(feature = "process_internals", issue = "none")]

#[cfg(test)]
mod tests;

use crate::cmp;
use crate::collections::BTreeMap;
use crate::convert::{TryFrom, TryInto};
use crate::env;
use crate::env::consts::{EXE_EXTENSION, EXE_SUFFIX};
use crate::ffi::{OsStr, OsString};
use crate::fmt;
use crate::io::{self, Error, ErrorKind};
use crate::mem;
use crate::num::NonZeroI32;
use crate::os::windows::ffi::{OsStrExt, OsStringExt};
use crate::os::windows::io::{AsRawHandle, FromRawHandle, IntoRawHandle};
use crate::path::{Path, PathBuf};
use crate::ptr;
use crate::sys::c;
use crate::sys::c::NonZeroDWORD;
use crate::sys::fs::{File, OpenOptions};
use crate::sys::handle::Handle;
use crate::sys::path;
use crate::sys::pipe::{self, AnonPipe};
use crate::sys::stdio;
use crate::sys::{cvt, to_u16s};
use crate::sys_common::mutex::StaticMutex;
use crate::sys_common::process::{CommandEnv, CommandEnvs};
use crate::sys_common::{AsInner, IntoInner};

use libc::{c_void, EXIT_FAILURE, EXIT_SUCCESS};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Eq)]
#[doc(hidden)]
pub struct EnvKey {
    os_string: OsString,
    // This stores a UTF-16 encoded string to workaround the mismatch between
    // Rust's OsString (WTF-8) and the Windows API string type (UTF-16).
    // Normally converting on every API call is acceptable but here
    // `c::CompareStringOrdinal` will be called for every use of `==`.
    utf16: Vec<u16>,
}

impl EnvKey {
    fn new<T: Into<OsString>>(key: T) -> Self {
        EnvKey::from(key.into())
    }
}

// Comparing Windows environment variable keys[1] are behaviourally the
// composition of two operations[2]:
//
// 1. Case-fold both strings. This is done using a language-independent
// uppercase mapping that's unique to Windows (albeit based on data from an
// older Unicode spec). It only operates on individual UTF-16 code units so
// surrogates are left unchanged. This uppercase mapping can potentially change
// between Windows versions.
//
// 2. Perform an ordinal comparison of the strings. A comparison using ordinal
// is just a comparison based on the numerical value of each UTF-16 code unit[3].
//
// Because the case-folding mapping is unique to Windows and not guaranteed to
// be stable, we ask the OS to compare the strings for us. This is done by
// calling `CompareStringOrdinal`[4] with `bIgnoreCase` set to `TRUE`.
//
// [1] https://docs.microsoft.com/en-us/dotnet/standard/base-types/best-practices-strings#choosing-a-stringcomparison-member-for-your-method-call
// [2] https://docs.microsoft.com/en-us/dotnet/standard/base-types/best-practices-strings#stringtoupper-and-stringtolower
// [3] https://docs.microsoft.com/en-us/dotnet/api/system.stringcomparison?view=net-5.0#System_StringComparison_Ordinal
// [4] https://docs.microsoft.com/en-us/windows/win32/api/stringapiset/nf-stringapiset-comparestringordinal
impl Ord for EnvKey {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        unsafe {
            let result = c::CompareStringOrdinal(
                self.utf16.as_ptr(),
                self.utf16.len() as _,
                other.utf16.as_ptr(),
                other.utf16.len() as _,
                c::TRUE,
            );
            match result {
                c::CSTR_LESS_THAN => cmp::Ordering::Less,
                c::CSTR_EQUAL => cmp::Ordering::Equal,
                c::CSTR_GREATER_THAN => cmp::Ordering::Greater,
                // `CompareStringOrdinal` should never fail so long as the parameters are correct.
                _ => panic!("comparing environment keys failed: {}", Error::last_os_error()),
            }
        }
    }
}
impl PartialOrd for EnvKey {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for EnvKey {
    fn eq(&self, other: &Self) -> bool {
        if self.utf16.len() != other.utf16.len() {
            false
        } else {
            self.cmp(other) == cmp::Ordering::Equal
        }
    }
}
impl PartialOrd<str> for EnvKey {
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        Some(self.cmp(&EnvKey::new(other)))
    }
}
impl PartialEq<str> for EnvKey {
    fn eq(&self, other: &str) -> bool {
        if self.os_string.len() != other.len() {
            false
        } else {
            self.cmp(&EnvKey::new(other)) == cmp::Ordering::Equal
        }
    }
}

// Environment variable keys should preserve their original case even though
// they are compared using a caseless string mapping.
impl From<OsString> for EnvKey {
    fn from(k: OsString) -> Self {
        EnvKey { utf16: k.encode_wide().collect(), os_string: k }
    }
}

impl From<EnvKey> for OsString {
    fn from(k: EnvKey) -> Self {
        k.os_string
    }
}

impl From<&OsStr> for EnvKey {
    fn from(k: &OsStr) -> Self {
        Self::from(k.to_os_string())
    }
}

impl AsRef<OsStr> for EnvKey {
    fn as_ref(&self) -> &OsStr {
        &self.os_string
    }
}

fn ensure_no_nuls<T: AsRef<OsStr>>(str: T) -> io::Result<T> {
    if str.as_ref().encode_wide().any(|b| b == 0) {
        Err(io::Error::new_const(ErrorKind::InvalidInput, &"nul byte found in provided data"))
    } else {
        Ok(str)
    }
}

pub struct Command {
    program: OsString,
    args: Vec<Arg>,
    env: CommandEnv,
    cwd: Option<OsString>,
    flags: u32,
    detach: bool, // not currently exposed in std::process
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
    force_quotes_enabled: bool,
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

#[derive(Debug)]
enum Arg {
    /// Add quotes (if needed)
    Regular(OsString),
    /// Append raw string without quoting
    Raw(OsString),
}

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
            force_quotes_enabled: false,
        }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(Arg::Regular(arg.to_os_string()))
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

    pub fn force_quotes(&mut self, enabled: bool) {
        self.force_quotes_enabled = enabled;
    }

    pub fn raw_arg(&mut self, command_str_to_append: &OsStr) {
        self.args.push(Arg::Raw(command_str_to_append.to_os_string()))
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
        let maybe_env = self.env.capture_if_changed();

        let mut si = zeroed_startupinfo();
        si.cb = mem::size_of::<c::STARTUPINFO>() as c::DWORD;
        si.dwFlags = c::STARTF_USESTDHANDLES;

        let child_paths = if let Some(env) = maybe_env.as_ref() {
            env.get(&EnvKey::new("PATH")).map(|s| s.as_os_str())
        } else {
            None
        };
        let program = resolve_exe(&self.program, child_paths)?;
        let mut cmd_str =
            make_command_line(program.as_os_str(), &self.args, self.force_quotes_enabled)?;
        cmd_str.push(0); // add null terminator

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
        // https://support.microsoft.com/kb/315939
        static CREATE_PROCESS_LOCK: StaticMutex = StaticMutex::new();

        let _guard = unsafe { CREATE_PROCESS_LOCK.lock() };

        let mut pipes = StdioPipes { stdin: None, stdout: None, stderr: None };
        let null = Stdio::Null;
        let default_stdin = if needs_stdin { &default } else { &null };
        let stdin = self.stdin.as_ref().unwrap_or(default_stdin);
        let stdout = self.stdout.as_ref().unwrap_or(&default);
        let stderr = self.stderr.as_ref().unwrap_or(&default);
        let stdin = stdin.to_handle(c::STD_INPUT_HANDLE, &mut pipes.stdin)?;
        let stdout = stdout.to_handle(c::STD_OUTPUT_HANDLE, &mut pipes.stdout)?;
        let stderr = stderr.to_handle(c::STD_ERROR_HANDLE, &mut pipes.stderr)?;
        si.hStdInput = stdin.as_raw_handle();
        si.hStdOutput = stdout.as_raw_handle();
        si.hStdError = stderr.as_raw_handle();

        let program = to_u16s(&program)?;
        unsafe {
            cvt(c::CreateProcessW(
                program.as_ptr(),
                cmd_str.as_mut_ptr(),
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
        unsafe {
            drop(Handle::from_raw_handle(pi.hThread));

            Ok((Process { handle: Handle::from_raw_handle(pi.hProcess) }, pipes))
        }
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.program.fmt(f)?;
        for arg in &self.args {
            f.write_str(" ")?;
            match arg {
                Arg::Regular(s) => s.fmt(f),
                Arg::Raw(s) => f.write_str(&s.to_string_lossy()),
            }?;
        }
        Ok(())
    }
}

// Resolve `exe_path` to the executable name.
//
// * If the path is simply a file name then use the paths given by `search_paths` to find the executable.
// * Otherwise use the `exe_path` as given.
//
// This function may also append `.exe` to the name. The rationale for doing so is as follows:
//
// It is a very strong convention that Windows executables have the `exe` extension.
// In Rust, it is common to omit this extension.
// Therefore this functions first assumes `.exe` was intended.
// It falls back to the plain file name if a full path is given and the extension is omitted
// or if only a file name is given and it already contains an extension.
fn resolve_exe<'a>(exe_path: &'a OsStr, child_paths: Option<&OsStr>) -> io::Result<PathBuf> {
    // Early return if there is no filename.
    if exe_path.is_empty() || path::has_trailing_slash(exe_path) {
        return Err(io::Error::new_const(
            io::ErrorKind::InvalidInput,
            &"program path has no file name",
        ));
    }
    // Test if the file name has the `exe` extension.
    // This does a case-insensitive `ends_with`.
    let has_exe_suffix = if exe_path.len() >= EXE_SUFFIX.len() {
        exe_path.bytes()[exe_path.len() - EXE_SUFFIX.len()..]
            .eq_ignore_ascii_case(EXE_SUFFIX.as_bytes())
    } else {
        false
    };

    // If `exe_path` is an absolute path or a sub-path then don't search `PATH` for it.
    if !path::is_file_name(exe_path) {
        if has_exe_suffix {
            // The application name is a path to a `.exe` file.
            // Let `CreateProcessW` figure out if it exists or not.
            return Ok(exe_path.into());
        }
        let mut path = PathBuf::from(exe_path);

        // Append `.exe` if not already there.
        path = path::append_suffix(path, EXE_SUFFIX.as_ref());
        if path.try_exists().unwrap_or(false) {
            return Ok(path);
        } else {
            // It's ok to use `set_extension` here because the intent is to
            // remove the extension that was just added.
            path.set_extension("");
            return Ok(path);
        }
    } else {
        ensure_no_nuls(exe_path)?;
        // From the `CreateProcessW` docs:
        // > If the file name does not contain an extension, .exe is appended.
        // Note that this rule only applies when searching paths.
        let has_extension = exe_path.bytes().contains(&b'.');

        // Search the directories given by `search_paths`.
        let result = search_paths(child_paths, |mut path| {
            path.push(&exe_path);
            if !has_extension {
                path.set_extension(EXE_EXTENSION);
            }
            if let Ok(true) = path.try_exists() { Some(path) } else { None }
        });
        if let Some(path) = result {
            return Ok(path);
        }
    }
    // If we get here then the executable cannot be found.
    Err(io::Error::new_const(io::ErrorKind::NotFound, &"program not found"))
}

// Calls `f` for every path that should be used to find an executable.
// Returns once `f` returns the path to an executable or all paths have been searched.
fn search_paths<F>(child_paths: Option<&OsStr>, mut f: F) -> Option<PathBuf>
where
    F: FnMut(PathBuf) -> Option<PathBuf>,
{
    // 1. Child paths
    // This is for consistency with Rust's historic behaviour.
    if let Some(paths) = child_paths {
        for path in env::split_paths(paths).filter(|p| !p.as_os_str().is_empty()) {
            if let Some(path) = f(path) {
                return Some(path);
            }
        }
    }

    // 2. Application path
    if let Ok(mut app_path) = env::current_exe() {
        app_path.pop();
        if let Some(path) = f(app_path) {
            return Some(path);
        }
    }

    // 3 & 4. System paths
    // SAFETY: This uses `fill_utf16_buf` to safely call the OS functions.
    unsafe {
        if let Ok(Some(path)) = super::fill_utf16_buf(
            |buf, size| c::GetSystemDirectoryW(buf, size),
            |buf| f(PathBuf::from(OsString::from_wide(buf))),
        ) {
            return Some(path);
        }
        #[cfg(not(target_vendor = "uwp"))]
        {
            if let Ok(Some(path)) = super::fill_utf16_buf(
                |buf, size| c::GetWindowsDirectoryW(buf, size),
                |buf| f(PathBuf::from(OsString::from_wide(buf))),
            ) {
                return Some(path);
            }
        }
    }

    // 5. Parent paths
    if let Some(parent_paths) = env::var_os("PATH") {
        for path in env::split_paths(&parent_paths).filter(|p| !p.as_os_str().is_empty()) {
            if let Some(path) = f(path) {
                return Some(path);
            }
        }
    }
    None
}

impl Stdio {
    fn to_handle(&self, stdio_id: c::DWORD, pipe: &mut Option<AnonPipe>) -> io::Result<Handle> {
        match *self {
            // If no stdio handle is available, then inherit means that it
            // should still be unavailable so propagate the
            // INVALID_HANDLE_VALUE.
            Stdio::Inherit => match stdio::get_handle(stdio_id) {
                Ok(io) => unsafe {
                    let io = Handle::from_raw_handle(io);
                    let ret = io.duplicate(0, true, c::DUPLICATE_SAME_ACCESS);
                    io.into_raw_handle();
                    ret
                },
                Err(..) => unsafe { Ok(Handle::from_raw_handle(c::INVALID_HANDLE_VALUE)) },
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
                File::open(Path::new("NUL"), &opts).map(|file| file.into_inner())
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
        Stdio::Handle(file.into_inner())
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
        cvt(unsafe { c::TerminateProcess(self.handle.as_raw_handle(), 1) })?;
        Ok(())
    }

    pub fn id(&self) -> u32 {
        unsafe { c::GetProcessId(self.handle.as_raw_handle()) as u32 }
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        unsafe {
            let res = c::WaitForSingleObject(self.handle.as_raw_handle(), c::INFINITE);
            if res != c::WAIT_OBJECT_0 {
                return Err(Error::last_os_error());
            }
            let mut status = 0;
            cvt(c::GetExitCodeProcess(self.handle.as_raw_handle(), &mut status))?;
            Ok(ExitStatus(status))
        }
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        unsafe {
            match c::WaitForSingleObject(self.handle.as_raw_handle(), 0) {
                c::WAIT_OBJECT_0 => {}
                c::WAIT_TIMEOUT => {
                    return Ok(None);
                }
                _ => return Err(io::Error::last_os_error()),
            }
            let mut status = 0;
            cvt(c::GetExitCodeProcess(self.handle.as_raw_handle(), &mut status))?;
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
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        match NonZeroDWORD::try_from(self.0) {
            /* was nonzero */ Ok(failure) => Err(ExitStatusError(failure)),
            /* was zero, couldn't convert */ Err(_) => Ok(()),
        }
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
pub struct ExitStatusError(c::NonZeroDWORD);

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        ExitStatus(self.0.into())
    }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZeroI32> {
        Some((u32::from(self.0) as i32).try_into().unwrap())
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

enum Quote {
    // Every arg is quoted
    Always,
    // Whitespace and empty args are quoted
    Auto,
    // Arg appended without any changes (#29494)
    Never,
}

// Produces a wide string *without terminating null*; returns an error if
// `prog` or any of the `args` contain a nul.
fn make_command_line(prog: &OsStr, args: &[Arg], force_quotes: bool) -> io::Result<Vec<u16>> {
    // Encode the command and arguments in a command line string such
    // that the spawned process may recover them using CommandLineToArgvW.
    let mut cmd: Vec<u16> = Vec::new();

    // CreateFileW has special handling for .bat and .cmd files, which means we
    // need to add an extra pair of quotes surrounding the whole command line
    // so they are properly passed on to the script.
    // See issue #91991.
    let is_batch_file = Path::new(prog)
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("cmd") || ext.eq_ignore_ascii_case("bat"))
        .unwrap_or(false);
    if is_batch_file {
        cmd.push(b'"' as u16);
    }

    // Always quote the program name so CreateProcess doesn't interpret args as
    // part of the name if the binary wasn't found first time.
    append_arg(&mut cmd, prog, Quote::Always)?;
    for arg in args {
        cmd.push(' ' as u16);
        let (arg, quote) = match arg {
            Arg::Regular(arg) => (arg, if force_quotes { Quote::Always } else { Quote::Auto }),
            Arg::Raw(arg) => (arg, Quote::Never),
        };
        append_arg(&mut cmd, arg, quote)?;
    }
    if is_batch_file {
        cmd.push(b'"' as u16);
    }
    return Ok(cmd);

    fn append_arg(cmd: &mut Vec<u16>, arg: &OsStr, quote: Quote) -> io::Result<()> {
        // If an argument has 0 characters then we need to quote it to ensure
        // that it actually gets passed through on the command line or otherwise
        // it will be dropped entirely when parsed on the other end.
        ensure_no_nuls(arg)?;
        let arg_bytes = &arg.as_inner().inner.as_inner();
        let (quote, escape) = match quote {
            Quote::Always => (true, true),
            Quote::Auto => {
                (arg_bytes.iter().any(|c| *c == b' ' || *c == b'\t') || arg_bytes.is_empty(), true)
            }
            Quote::Never => (false, false),
        };
        if quote {
            cmd.push('"' as u16);
        }

        let mut backslashes: usize = 0;
        for x in arg.encode_wide() {
            if escape {
                if x == '\\' as u16 {
                    backslashes += 1;
                } else {
                    if x == '"' as u16 {
                        // Add n+1 backslashes to total 2n+1 before internal '"'.
                        cmd.extend((0..=backslashes).map(|_| '\\' as u16));
                    }
                    backslashes = 0;
                }
            }
            cmd.push(x);
        }

        if quote {
            // Add n backslashes to total 2n before ending '"'.
            cmd.extend((0..backslashes).map(|_| '\\' as u16));
            cmd.push('"' as u16);
        }
        Ok(())
    }
}

fn make_envp(maybe_env: Option<BTreeMap<EnvKey, OsString>>) -> io::Result<(*mut c_void, Vec<u16>)> {
    // On Windows we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    if let Some(env) = maybe_env {
        let mut blk = Vec::new();

        // If there are no environment variables to set then signal this by
        // pushing a null.
        if env.is_empty() {
            blk.push(0);
        }

        for (k, v) in env {
            ensure_no_nuls(k.os_string)?;
            blk.extend(k.utf16);
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
    iter: crate::slice::Iter<'a, Arg>,
}

impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;
    fn next(&mut self) -> Option<&'a OsStr> {
        self.iter.next().map(|arg| match arg {
            Arg::Regular(s) | Arg::Raw(s) => s.as_ref(),
        })
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
