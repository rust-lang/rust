// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ascii::*;
use collections::HashMap;
use collections;
use env::split_paths;
use env;
use ffi::{OsString, OsStr};
use fmt;
use fs;
use io::{self, Error, ErrorKind};
use libc::c_void;
use mem;
use os::windows::ffi::OsStrExt;
use path::Path;
use ptr;
use sys::mutex::Mutex;
use sys::c;
use sys::fs::{OpenOptions, File};
use sys::handle::Handle;
use sys::pipe::{self, AnonPipe};
use sys::stdio;
use sys::{self, cvt};
use sys_common::{AsInner, FromInner};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

fn mk_key(s: &OsStr) -> OsString {
    FromInner::from_inner(sys::os_str::Buf {
        inner: s.as_inner().inner.to_ascii_uppercase()
    })
}

fn ensure_no_nuls<T: AsRef<OsStr>>(str: T) -> io::Result<T> {
    if str.as_ref().encode_wide().any(|b| b == 0) {
        Err(io::Error::new(ErrorKind::InvalidInput, "nul byte found in provided data"))
    } else {
        Ok(str)
    }
}

pub struct Command {
    program: OsString,
    args: Vec<OsString>,
    env: Option<HashMap<OsString, OsString>>,
    cwd: Option<OsString>,
    flags: u32,
    detach: bool, // not currently exposed in std::process
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
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

impl Command {
    pub fn new(program: &OsStr) -> Command {
        Command {
            program: program.to_os_string(),
            args: Vec::new(),
            env: None,
            cwd: None,
            flags: 0,
            detach: false,
            stdin: None,
            stdout: None,
            stderr: None,
        }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(arg.to_os_string())
    }
    fn init_env_map(&mut self){
        if self.env.is_none() {
            self.env = Some(env::vars_os().map(|(key, val)| {
                (mk_key(&key), val)
            }).collect());
        }
    }
    pub fn env(&mut self, key: &OsStr, val: &OsStr) {
        self.init_env_map();
        self.env.as_mut().unwrap().insert(mk_key(key), val.to_os_string());
    }
    pub fn env_remove(&mut self, key: &OsStr) {
        self.init_env_map();
        self.env.as_mut().unwrap().remove(&mk_key(key));
    }
    pub fn env_clear(&mut self) {
        self.env = Some(HashMap::new())
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

    pub fn spawn(&mut self, default: Stdio, needs_stdin: bool)
                 -> io::Result<(Process, StdioPipes)> {
        // To have the spawning semantics of unix/windows stay the same, we need
        // to read the *child's* PATH if one is provided. See #15149 for more
        // details.
        let program = self.env.as_ref().and_then(|env| {
            for (key, v) in env {
                if OsStr::new("PATH") != &**key { continue }

                // Split the value and test each path to see if the
                // program exists.
                for path in split_paths(&v) {
                    let path = path.join(self.program.to_str().unwrap())
                                   .with_extension(env::consts::EXE_EXTENSION);
                    if fs::metadata(&path).is_ok() {
                        return Some(path.into_os_string())
                    }
                }
                break
            }
            None
        });

        let mut si = zeroed_startupinfo();
        si.cb = mem::size_of::<c::STARTUPINFO>() as c::DWORD;
        si.dwFlags = c::STARTF_USESTDHANDLES;

        let program = program.as_ref().unwrap_or(&self.program);
        let mut cmd_str = make_command_line(program, &self.args)?;
        cmd_str.push(0); // add null terminator

        // stolen from the libuv code.
        let mut flags = self.flags | c::CREATE_UNICODE_ENVIRONMENT;
        if self.detach {
            flags |= c::DETACHED_PROCESS | c::CREATE_NEW_PROCESS_GROUP;
        }

        let (envp, _data) = make_envp(self.env.as_ref())?;
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

        let mut pipes = StdioPipes {
            stdin: None,
            stdout: None,
            stderr: None,
        };
        let null = Stdio::Null;
        let default_stdin = if needs_stdin {&default} else {&null};
        let stdin = self.stdin.as_ref().unwrap_or(default_stdin);
        let stdout = self.stdout.as_ref().unwrap_or(&default);
        let stderr = self.stderr.as_ref().unwrap_or(&default);
        let stdin = stdin.to_handle(c::STD_INPUT_HANDLE, &mut pipes.stdin)?;
        let stdout = stdout.to_handle(c::STD_OUTPUT_HANDLE,
                                      &mut pipes.stdout)?;
        let stderr = stderr.to_handle(c::STD_ERROR_HANDLE,
                                      &mut pipes.stderr)?;
        si.hStdInput = stdin.raw();
        si.hStdOutput = stdout.raw();
        si.hStdError = stderr.raw();

        unsafe {
            cvt(c::CreateProcessW(ptr::null(),
                                  cmd_str.as_mut_ptr(),
                                  ptr::null_mut(),
                                  ptr::null_mut(),
                                  c::TRUE, flags, envp, dirp,
                                  &mut si, &mut pi))
        }?;

        // We close the thread handle because we don't care about keeping
        // the thread id valid, and we aren't keeping the thread handle
        // around to be able to close it later.
        drop(Handle::new(pi.hThread));

        Ok((Process { handle: Handle::new(pi.hProcess) }, pipes))
    }

}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
            DropGuard { lock: lock }
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
    fn to_handle(&self, stdio_id: c::DWORD, pipe: &mut Option<AnonPipe>)
                 -> io::Result<Handle> {
        match *self {
            // If no stdio handle is available, then inherit means that it
            // should still be unavailable so propagate the
            // INVALID_HANDLE_VALUE.
            Stdio::Inherit => {
                match stdio::get(stdio_id) {
                    Ok(io) => io.handle().duplicate(0, true,
                                                    c::DUPLICATE_SAME_ACCESS),
                    Err(..) => Ok(Handle::new(c::INVALID_HANDLE_VALUE)),
                }
            }

            Stdio::MakePipe => {
                let ours_readable = stdio_id != c::STD_INPUT_HANDLE;
                let pipes = pipe::anon_pipe(ours_readable)?;
                *pipe = Some(pipes.ours);
                cvt(unsafe {
                    c::SetHandleInformation(pipes.theirs.handle().raw(),
                                            c::HANDLE_FLAG_INHERIT,
                                            c::HANDLE_FLAG_INHERIT)
                })?;
                Ok(pipes.theirs.into_handle())
            }

            Stdio::Handle(ref handle) => {
                handle.duplicate(0, true, c::DUPLICATE_SAME_ACCESS)
            }

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
                File::open(Path::new("NUL"), &opts).map(|file| {
                    file.into_handle()
                })
            }
        }
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
        cvt(unsafe {
            c::TerminateProcess(self.handle.raw(), 1)
        })?;
        Ok(())
    }

    pub fn id(&self) -> u32 {
        unsafe {
            c::GetProcessId(self.handle.raw()) as u32
        }
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        unsafe {
            let res = c::WaitForSingleObject(self.handle.raw(), c::INFINITE);
            if res != c::WAIT_OBJECT_0 {
                return Err(Error::last_os_error())
            }
            let mut status = 0;
            cvt(c::GetExitCodeProcess(self.handle.raw(), &mut status))?;
            Ok(ExitStatus(status))
        }
    }

    pub fn try_wait(&mut self) -> io::Result<ExitStatus> {
        unsafe {
            match c::WaitForSingleObject(self.handle.raw(), 0) {
                c::WAIT_OBJECT_0 => {}
                c::WAIT_TIMEOUT => {
                    return Err(io::Error::from_raw_os_error(c::WSAEWOULDBLOCK))
                }
                _ => return Err(io::Error::last_os_error()),
            }
            let mut status = 0;
            cvt(c::GetExitCodeProcess(self.handle.raw(), &mut status))?;
            Ok(ExitStatus(status))
        }
    }

    pub fn handle(&self) -> &Handle { &self.handle }

    pub fn into_handle(self) -> Handle { self.handle }
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

impl From<c::DWORD> for ExitStatus {
    fn from(u: c::DWORD) -> ExitStatus {
        ExitStatus(u)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "exit code: {}", self.0)
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
        dwThreadId: 0
    }
}

// Produces a wide string *without terminating null*; returns an error if
// `prog` or any of the `args` contain a nul.
fn make_command_line(prog: &OsStr, args: &[OsString]) -> io::Result<Vec<u16>> {
    // Encode the command and arguments in a command line string such
    // that the spawned process may recover them using CommandLineToArgvW.
    let mut cmd: Vec<u16> = Vec::new();
    append_arg(&mut cmd, prog)?;
    for arg in args {
        cmd.push(' ' as u16);
        append_arg(&mut cmd, arg)?;
    }
    return Ok(cmd);

    fn append_arg(cmd: &mut Vec<u16>, arg: &OsStr) -> io::Result<()> {
        // If an argument has 0 characters then we need to quote it to ensure
        // that it actually gets passed through on the command line or otherwise
        // it will be dropped entirely when parsed on the other end.
        ensure_no_nuls(arg)?;
        let arg_bytes = &arg.as_inner().inner.as_inner();
        let quote = arg_bytes.iter().any(|c| *c == b' ' || *c == b'\t')
            || arg_bytes.is_empty();
        if quote {
            cmd.push('"' as u16);
        }

        let mut iter = arg.encode_wide();
        let mut backslashes: usize = 0;
        while let Some(x) = iter.next() {
            if x == '\\' as u16 {
                backslashes += 1;
            } else {
                if x == '"' as u16 {
                    // Add n+1 backslashes to total 2n+1 before internal '"'.
                    for _ in 0..(backslashes+1) {
                        cmd.push('\\' as u16);
                    }
                }
                backslashes = 0;
            }
            cmd.push(x);
        }

        if quote {
            // Add n backslashes to total 2n before ending '"'.
            for _ in 0..backslashes {
                cmd.push('\\' as u16);
            }
            cmd.push('"' as u16);
        }
        Ok(())
    }
}

fn make_envp(env: Option<&collections::HashMap<OsString, OsString>>)
             -> io::Result<(*mut c_void, Vec<u16>)> {
    // On Windows we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    match env {
        Some(env) => {
            let mut blk = Vec::new();

            for pair in env {
                blk.extend(ensure_no_nuls(pair.0)?.encode_wide());
                blk.push('=' as u16);
                blk.extend(ensure_no_nuls(pair.1)?.encode_wide());
                blk.push(0);
            }
            blk.push(0);
            Ok((blk.as_mut_ptr() as *mut c_void, blk))
        }
        _ => Ok((ptr::null_mut(), Vec::new()))
    }
}

fn make_dirp(d: Option<&OsString>) -> io::Result<(*const u16, Vec<u16>)> {

    match d {
        Some(dir) => {
            let mut dir_str: Vec<u16> = ensure_no_nuls(dir)?.encode_wide().collect();
            dir_str.push(0);
            Ok((dir_str.as_ptr(), dir_str))
        },
        None => Ok((ptr::null(), Vec::new()))
    }
}

#[cfg(test)]
mod tests {
    use ffi::{OsStr, OsString};
    use super::make_command_line;

    #[test]
    fn test_make_command_line() {
        fn test_wrapper(prog: &str, args: &[&str]) -> String {
            let command_line = &make_command_line(OsStr::new(prog),
                                                  &args.iter()
                                                       .map(|a| OsString::from(a))
                                                       .collect::<Vec<OsString>>())
                                    .unwrap();
            String::from_utf16(command_line).unwrap()
        }

        assert_eq!(
            test_wrapper("prog", &["aaa", "bbb", "ccc"]),
            "prog aaa bbb ccc"
        );

        assert_eq!(
            test_wrapper("C:\\Program Files\\blah\\blah.exe", &["aaa"]),
            "\"C:\\Program Files\\blah\\blah.exe\" aaa"
        );
        assert_eq!(
            test_wrapper("C:\\Program Files\\test", &["aa\"bb"]),
            "\"C:\\Program Files\\test\" aa\\\"bb"
        );
        assert_eq!(
            test_wrapper("echo", &["a b c"]),
            "echo \"a b c\""
        );
        assert_eq!(
            test_wrapper("echo", &["\" \\\" \\", "\\"]),
            "echo \"\\\" \\\\\\\" \\\\\" \\"
        );
        assert_eq!(
            test_wrapper("\u{03c0}\u{042f}\u{97f3}\u{00e6}\u{221e}", &[]),
            "\u{03c0}\u{042f}\u{97f3}\u{00e6}\u{221e}"
        );
    }
}
