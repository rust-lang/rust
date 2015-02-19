// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use ascii::*;
use collections::HashMap;
use collections;
use env;
use ffi::{OsString, OsStr};
use fmt;
use io::{self, Error};
use libc::{self, c_void};
use old_io::fs;
use old_path;
use os::windows::OsStrExt;
use ptr;
use sync::{StaticMutex, MUTEX_INIT};
use sys::pipe2::AnonPipe;
use sys::{self, cvt};
use sys::handle::Handle;
use sys_common::{AsInner, FromInner};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

fn mk_key(s: &OsStr) -> OsString {
    FromInner::from_inner(sys::os_str::Buf {
        inner: s.as_inner().inner.to_ascii_uppercase()
    })
}

#[derive(Clone)]
pub struct Command {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub env: Option<HashMap<OsString, OsString>>,
    pub cwd: Option<OsString>,
    pub detach: bool, // not currently exposed in std::process
}

impl Command {
    pub fn new(program: &OsStr) -> Command {
        Command {
            program: program.to_os_string(),
            args: Vec::new(),
            env: None,
            cwd: None,
            detach: false,
        }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(arg.to_os_string())
    }
    pub fn args<'a, I: Iterator<Item = &'a OsStr>>(&mut self, args: I) {
        self.args.extend(args.map(OsStr::to_os_string))
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
}

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

// `CreateProcess` is racy!
// http://support.microsoft.com/kb/315939
static CREATE_PROCESS_LOCK: StaticMutex = MUTEX_INIT;

/// A value representing a child process.
///
/// The lifetime of this value is linked to the lifetime of the actual
/// process - the Process destructor calls self.finish() which waits
/// for the process to terminate.
pub struct Process {
    /// A HANDLE to the process, which will prevent the pid being
    /// re-used until the handle is closed.
    handle: Handle,
}

impl Process {
    #[allow(deprecated)]
    pub fn spawn(cfg: &Command,
                 in_fd: Option<AnonPipe>, out_fd: Option<AnonPipe>, err_fd: Option<AnonPipe>)
                 -> io::Result<Process>
    {
        use libc::types::os::arch::extra::{DWORD, HANDLE, STARTUPINFO};
        use libc::consts::os::extra::{
            TRUE, FALSE,
            STARTF_USESTDHANDLES,
            INVALID_HANDLE_VALUE,
            DUPLICATE_SAME_ACCESS
        };
        use libc::funcs::extra::kernel32::{
            GetCurrentProcess,
            DuplicateHandle,
            CloseHandle,
            CreateProcessW
        };

        use env::split_paths;
        use mem;
        use iter::IteratorExt;
        use str::StrExt;

        // To have the spawning semantics of unix/windows stay the same, we need to
        // read the *child's* PATH if one is provided. See #15149 for more details.
        let program = cfg.env.as_ref().and_then(|env| {
            for (key, v) in env {
                if OsStr::from_str("PATH") != &**key { continue }

                // Split the value and test each path to see if the
                // program exists.
                for path in split_paths(&v) {
                    let path = path.join(cfg.program.to_str().unwrap())
                                   .with_extension(env::consts::EXE_EXTENSION);
                    // FIXME: update with new fs module once it lands
                    if fs::stat(&old_path::Path::new(&path)).is_ok() {
                        return Some(OsString::from_str(path.as_str().unwrap()))
                    }
                }
                break
            }
            None
        });

        unsafe {
            let mut si = zeroed_startupinfo();
            si.cb = mem::size_of::<STARTUPINFO>() as DWORD;
            si.dwFlags = STARTF_USESTDHANDLES;

            let cur_proc = GetCurrentProcess();

            // Similarly to unix, we don't actually leave holes for the stdio file
            // descriptors, but rather open up /dev/null equivalents. These
            // equivalents are drawn from libuv's windows process spawning.
            let set_fd = |&: fd: &Option<AnonPipe>, slot: &mut HANDLE,
                          is_stdin: bool| {
                match *fd {
                    None => {
                        let access = if is_stdin {
                            libc::FILE_GENERIC_READ
                        } else {
                            libc::FILE_GENERIC_WRITE | libc::FILE_READ_ATTRIBUTES
                        };
                        let size = mem::size_of::<libc::SECURITY_ATTRIBUTES>();
                        let mut sa = libc::SECURITY_ATTRIBUTES {
                            nLength: size as libc::DWORD,
                            lpSecurityDescriptor: ptr::null_mut(),
                            bInheritHandle: 1,
                        };
                        let mut filename: Vec<u16> = "NUL".utf16_units().collect();
                        filename.push(0);
                        *slot = libc::CreateFileW(filename.as_ptr(),
                                                  access,
                                                  libc::FILE_SHARE_READ |
                                                      libc::FILE_SHARE_WRITE,
                                                  &mut sa,
                                                  libc::OPEN_EXISTING,
                                                  0,
                                                  ptr::null_mut());
                        if *slot == INVALID_HANDLE_VALUE {
                            return Err(Error::last_os_error())
                        }
                    }
                    Some(ref pipe) => {
                        let orig = pipe.raw();
                        if orig == INVALID_HANDLE_VALUE {
                            return Err(Error::last_os_error())
                        }
                        if DuplicateHandle(cur_proc, orig, cur_proc, slot,
                                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
                            return Err(Error::last_os_error())
                        }
                    }
                }
                Ok(())
            };

            try!(set_fd(&in_fd, &mut si.hStdInput, true));
            try!(set_fd(&out_fd, &mut si.hStdOutput, false));
            try!(set_fd(&err_fd, &mut si.hStdError, false));

            let mut cmd_str = make_command_line(program.as_ref().unwrap_or(&cfg.program),
                                            &cfg.args);
            cmd_str.push(0); // add null terminator

            let mut pi = zeroed_process_information();
            let mut create_err = None;

            // stolen from the libuv code.
            let mut flags = libc::CREATE_UNICODE_ENVIRONMENT;
            if cfg.detach {
                flags |= libc::DETACHED_PROCESS | libc::CREATE_NEW_PROCESS_GROUP;
            }

            with_envp(cfg.env.as_ref(), |envp| {
                with_dirp(cfg.cwd.as_ref(), |dirp| {
                    let _lock = CREATE_PROCESS_LOCK.lock().unwrap();
                    let created = CreateProcessW(ptr::null(),
                                                 cmd_str.as_mut_ptr(),
                                                 ptr::null_mut(),
                                                 ptr::null_mut(),
                                                 TRUE,
                                                 flags, envp, dirp,
                                                 &mut si, &mut pi);
                    if created == FALSE {
                        create_err = Some(Error::last_os_error());
                    }
                })
            });

            assert!(CloseHandle(si.hStdInput) != 0);
            assert!(CloseHandle(si.hStdOutput) != 0);
            assert!(CloseHandle(si.hStdError) != 0);

            match create_err {
                Some(err) => return Err(err),
                None => {}
            }

            // We close the thread handle because we don't care about keeping the
            // thread id valid, and we aren't keeping the thread handle around to be
            // able to close it later. We don't close the process handle however
            // because std::we want the process id to stay valid at least until the
            // calling code closes the process handle.
            assert!(CloseHandle(pi.hThread) != 0);

            Ok(Process {
                handle: Handle::new(pi.hProcess)
            })
        }
    }

    pub unsafe fn kill(&self) -> io::Result<()> {
        try!(cvt(libc::TerminateProcess(self.handle.raw(), 1)));
        Ok(())
    }

    pub fn wait(&self) -> io::Result<ExitStatus> {
        use libc::consts::os::extra::{
            FALSE,
            STILL_ACTIVE,
            INFINITE,
            WAIT_OBJECT_0,
        };
        use libc::funcs::extra::kernel32::{
            GetExitCodeProcess,
            WaitForSingleObject,
        };

        unsafe {
            loop {
                let mut status = 0;
                if GetExitCodeProcess(self.handle.raw(), &mut status) == FALSE {
                    let err = Err(Error::last_os_error());
                    return err;
                }
                if status != STILL_ACTIVE {
                    return Ok(ExitStatus(status as i32));
                }
                match WaitForSingleObject(self.handle.raw(), INFINITE) {
                    WAIT_OBJECT_0 => {}
                    _ => {
                        let err = Err(Error::last_os_error());
                        return err
                    }
                }
            }
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(i32);

impl ExitStatus {
    pub fn success(&self) -> bool {
        self.0 == 0
    }
    pub fn code(&self) -> Option<i32> {
        Some(self.0)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "exit code: {}", self.0)
    }
}

fn zeroed_startupinfo() -> libc::types::os::arch::extra::STARTUPINFO {
    libc::types::os::arch::extra::STARTUPINFO {
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
        hStdInput: libc::INVALID_HANDLE_VALUE,
        hStdOutput: libc::INVALID_HANDLE_VALUE,
        hStdError: libc::INVALID_HANDLE_VALUE,
    }
}

fn zeroed_process_information() -> libc::types::os::arch::extra::PROCESS_INFORMATION {
    libc::types::os::arch::extra::PROCESS_INFORMATION {
        hProcess: ptr::null_mut(),
        hThread: ptr::null_mut(),
        dwProcessId: 0,
        dwThreadId: 0
    }
}

// Produces a wide string *without terminating null*
fn make_command_line(prog: &OsStr, args: &[OsString]) -> Vec<u16> {
    let mut cmd: Vec<u16> = Vec::new();
    append_arg(&mut cmd, prog);
    for arg in args {
        cmd.push(' ' as u16);
        append_arg(&mut cmd, arg);
    }
    return cmd;

    fn append_arg(cmd: &mut Vec<u16>, arg: &OsStr) {
        // If an argument has 0 characters then we need to quote it to ensure
        // that it actually gets passed through on the command line or otherwise
        // it will be dropped entirely when parsed on the other end.
        let arg_bytes = &arg.as_inner().inner.as_inner();
        let quote = arg_bytes.iter().any(|c| *c == b' ' || *c == b'\t')
            || arg_bytes.len() == 0;
        if quote {
            cmd.push('"' as u16);
        }

        let mut iter = arg.encode_wide();
        while let Some(x) = iter.next() {
            if x == '"' as u16 {
                // escape quotes
                cmd.push('\\' as u16);
                cmd.push('"' as u16);
            } else if x == '\\' as u16 {
                // is this a run of backslashes followed by a " ?
                if iter.clone().skip_while(|y| *y == '\\' as u16).next() == Some('"' as u16) {
                    // Double it ... NOTE: this behavior is being
                    // preserved as it's been part of Rust for a long
                    // time, but no one seems to know exactly why this
                    // is the right thing to do.
                    cmd.push('\\' as u16);
                    cmd.push('\\' as u16);
                } else {
                    // Push it through unescaped
                    cmd.push('\\' as u16);
                }
            } else {
                cmd.push(x)
            }
        }

        if quote {
            cmd.push('"' as u16);
        }
    }
}

fn with_envp<F, T>(env: Option<&collections::HashMap<OsString, OsString>>, cb: F) -> T
    where F: FnOnce(*mut c_void) -> T,
{
    // On Windows we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    match env {
        Some(env) => {
            let mut blk = Vec::new();

            for pair in env {
                blk.extend(pair.0.encode_wide());
                blk.push('=' as u16);
                blk.extend(pair.1.encode_wide());
                blk.push(0);
            }
            blk.push(0);
            cb(blk.as_mut_ptr() as *mut c_void)
        }
        _ => cb(ptr::null_mut())
    }
}

fn with_dirp<T, F>(d: Option<&OsString>, cb: F) -> T where
    F: FnOnce(*const u16) -> T,
{
    match d {
      Some(dir) => {
          let mut dir_str: Vec<u16> = dir.encode_wide().collect();
          dir_str.push(0);
          cb(dir_str.as_ptr())
      },
      None => cb(ptr::null())
    }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use str;
    use ffi::{OsStr, OsString};
    use super::make_command_line;

    #[test]
    fn test_make_command_line() {
        fn test_wrapper(prog: &str, args: &[&str]) -> String {
            String::from_utf16(
                &make_command_line(OsStr::from_str(prog),
                                   args.iter()
                                       .map(|a| OsString::from_str(a))
                                       .collect::<Vec<OsString>>()
                                       .as_slice())).unwrap()
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
            test_wrapper("\u{03c0}\u{042f}\u{97f3}\u{00e6}\u{221e}", &[..]),
            "\u{03c0}\u{042f}\u{97f3}\u{00e6}\u{221e}"
        );
    }
}
