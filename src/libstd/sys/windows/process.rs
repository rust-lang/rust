// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{pid_t, c_void, c_int};
use libc;
use c_str::CString;
use io;
use mem;
use os;
use ptr;
use prelude::*;
use io::process::{ProcessExit, ExitStatus, ExitSignal};
use collections;
use path::BytesContainer;
use hash::Hash;
use io::{IoResult, IoError};

use sys::fs;
use sys::{mod, retry, c, wouldblock, set_nonblocking, ms_to_timeval, timer};
use sys::fs::FileDesc;
use sys_common::helper_thread::Helper;
use sys_common::{AsInner, mkerr_libc, timeout};

use io::fs::PathExtensions;
use string::String;

pub use sys_common::ProcessConfig;

/// A value representing a child process.
///
/// The lifetime of this value is linked to the lifetime of the actual
/// process - the Process destructor calls self.finish() which waits
/// for the process to terminate.
pub struct Process {
    /// The unique id of the process (this should never be negative).
    pid: pid_t,

    /// A HANDLE to the process, which will prevent the pid being
    /// re-used until the handle is closed.
    handle: *mut (),
}

impl Drop for Process {
    fn drop(&mut self) {
        free_handle(self.handle);
    }
}

impl Process {
    pub fn id(&self) -> pid_t {
        self.pid
    }

    pub unsafe fn kill(&self, signal: int) -> IoResult<()> {
        Process::killpid(self.pid, signal)
    }

    pub unsafe fn killpid(pid: pid_t, signal: int) -> IoResult<()> {
        let handle = libc::OpenProcess(libc::PROCESS_TERMINATE |
                                       libc::PROCESS_QUERY_INFORMATION,
                                       libc::FALSE, pid as libc::DWORD);
        if handle.is_null() {
            return Err(super::last_error())
        }
        let ret = match signal {
            // test for existence on signal 0
            0 => {
                let mut status = 0;
                let ret = libc::GetExitCodeProcess(handle, &mut status);
                if ret == 0 {
                    Err(super::last_error())
                } else if status != libc::STILL_ACTIVE {
                    Err(IoError {
                        kind: io::InvalidInput,
                        desc: "no process to kill",
                        detail: None,
                    })
                } else {
                    Ok(())
                }
            }
            15 | 9 => { // sigterm or sigkill
                let ret = libc::TerminateProcess(handle, 1);
                super::mkerr_winbool(ret)
            }
            _ => Err(IoError {
                kind: io::IoUnavailable,
                desc: "unsupported signal on windows",
                detail: None,
            })
        };
        let _ = libc::CloseHandle(handle);
        return ret;
    }

    pub fn spawn<K, V, C, P>(cfg: &C, in_fd: Option<P>,
                              out_fd: Option<P>, err_fd: Option<P>)
                              -> IoResult<Process>
        where C: ProcessConfig<K, V>, P: AsInner<FileDesc>,
              K: BytesContainer + Eq + Hash, V: BytesContainer
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
        use libc::funcs::extra::msvcrt::get_osfhandle;

        use mem;
        use iter::{Iterator, IteratorExt};
        use str::StrPrelude;

        if cfg.gid().is_some() || cfg.uid().is_some() {
            return Err(IoError {
                kind: io::IoUnavailable,
                desc: "unsupported gid/uid requested on windows",
                detail: None,
            })
        }

        // To have the spawning semantics of unix/windows stay the same, we need to
        // read the *child's* PATH if one is provided. See #15149 for more details.
        let program = cfg.env().and_then(|env| {
            for (key, v) in env.iter() {
                if b"PATH" != key.container_as_bytes() { continue }

                // Split the value and test each path to see if the
                // program exists.
                for path in os::split_paths(v.container_as_bytes()).into_iter() {
                    let path = path.join(cfg.program().as_bytes_no_nul())
                                   .with_extension(os::consts::EXE_EXTENSION);
                    if path.exists() {
                        return Some(path.to_c_str())
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
            let set_fd = |fd: &Option<P>, slot: &mut HANDLE,
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
                            return Err(super::last_error())
                        }
                    }
                    Some(ref fd) => {
                        let orig = get_osfhandle(fd.as_inner().fd()) as HANDLE;
                        if orig == INVALID_HANDLE_VALUE {
                            return Err(super::last_error())
                        }
                        if DuplicateHandle(cur_proc, orig, cur_proc, slot,
                                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
                            return Err(super::last_error())
                        }
                    }
                }
                Ok(())
            };

            try!(set_fd(&in_fd, &mut si.hStdInput, true));
            try!(set_fd(&out_fd, &mut si.hStdOutput, false));
            try!(set_fd(&err_fd, &mut si.hStdError, false));

            let cmd_str = make_command_line(program.as_ref().unwrap_or(cfg.program()),
                                            cfg.args());
            let mut pi = zeroed_process_information();
            let mut create_err = None;

            // stolen from the libuv code.
            let mut flags = libc::CREATE_UNICODE_ENVIRONMENT;
            if cfg.detach() {
                flags |= libc::DETACHED_PROCESS | libc::CREATE_NEW_PROCESS_GROUP;
            }

            with_envp(cfg.env(), |envp| {
                with_dirp(cfg.cwd(), |dirp| {
                    let mut cmd_str: Vec<u16> = cmd_str.utf16_units().collect();
                    cmd_str.push(0);
                    let created = CreateProcessW(ptr::null(),
                                                 cmd_str.as_mut_ptr(),
                                                 ptr::null_mut(),
                                                 ptr::null_mut(),
                                                 TRUE,
                                                 flags, envp, dirp,
                                                 &mut si, &mut pi);
                    if created == FALSE {
                        create_err = Some(super::last_error());
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
                pid: pi.dwProcessId as pid_t,
                handle: pi.hProcess as *mut ()
            })
        }
    }

    /// Waits for a process to exit and returns the exit code, failing
    /// if there is no process with the specified id.
    ///
    /// Note that this is private to avoid race conditions on unix where if
    /// a user calls waitpid(some_process.get_id()) then some_process.finish()
    /// and some_process.destroy() and some_process.finalize() will then either
    /// operate on a none-existent process or, even worse, on a newer process
    /// with the same id.
    pub fn wait(&self, deadline: u64) -> IoResult<ProcessExit> {
        use libc::types::os::arch::extra::DWORD;
        use libc::consts::os::extra::{
            SYNCHRONIZE,
            PROCESS_QUERY_INFORMATION,
            FALSE,
            STILL_ACTIVE,
            INFINITE,
            WAIT_TIMEOUT,
            WAIT_OBJECT_0,
        };
        use libc::funcs::extra::kernel32::{
            OpenProcess,
            GetExitCodeProcess,
            CloseHandle,
            WaitForSingleObject,
        };

        unsafe {
            let process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_INFORMATION,
                                      FALSE,
                                      self.pid as DWORD);
            if process.is_null() {
                return Err(super::last_error())
            }

            loop {
                let mut status = 0;
                if GetExitCodeProcess(process, &mut status) == FALSE {
                    let err = Err(super::last_error());
                    assert!(CloseHandle(process) != 0);
                    return err;
                }
                if status != STILL_ACTIVE {
                    assert!(CloseHandle(process) != 0);
                    return Ok(ExitStatus(status as int));
                }
                let interval = if deadline == 0 {
                    INFINITE
                } else {
                    let now = timer::now();
                    if deadline < now {0} else {(deadline - now) as u32}
                };
                match WaitForSingleObject(process, interval) {
                    WAIT_OBJECT_0 => {}
                    WAIT_TIMEOUT => {
                        assert!(CloseHandle(process) != 0);
                        return Err(timeout("process wait timed out"))
                    }
                    _ => {
                        let err = Err(super::last_error());
                        assert!(CloseHandle(process) != 0);
                        return err
                    }
                }
            }
        }
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

fn make_command_line(prog: &CString, args: &[CString]) -> String {
    let mut cmd = String::new();
    append_arg(&mut cmd, prog.as_str()
                             .expect("expected program name to be utf-8 encoded"));
    for arg in args.iter() {
        cmd.push(' ');
        append_arg(&mut cmd, arg.as_str()
                                .expect("expected argument to be utf-8 encoded"));
    }
    return cmd;

    fn append_arg(cmd: &mut String, arg: &str) {
        // If an argument has 0 characters then we need to quote it to ensure
        // that it actually gets passed through on the command line or otherwise
        // it will be dropped entirely when parsed on the other end.
        let quote = arg.chars().any(|c| c == ' ' || c == '\t') || arg.len() == 0;
        if quote {
            cmd.push('"');
        }
        let argvec: Vec<char> = arg.chars().collect();
        for i in range(0u, argvec.len()) {
            append_char_at(cmd, argvec.as_slice(), i);
        }
        if quote {
            cmd.push('"');
        }
    }

    fn append_char_at(cmd: &mut String, arg: &[char], i: uint) {
        match arg[i] {
            '"' => {
                // Escape quotes.
                cmd.push_str("\\\"");
            }
            '\\' => {
                if backslash_run_ends_in_quote(arg, i) {
                    // Double all backslashes that are in runs before quotes.
                    cmd.push_str("\\\\");
                } else {
                    // Pass other backslashes through unescaped.
                    cmd.push('\\');
                }
            }
            c => {
                cmd.push(c);
            }
        }
    }

    fn backslash_run_ends_in_quote(s: &[char], mut i: uint) -> bool {
        while i < s.len() && s[i] == '\\' {
            i += 1;
        }
        return i < s.len() && s[i] == '"';
    }
}

fn with_envp<K, V, T>(env: Option<&collections::HashMap<K, V>>,
                      cb: |*mut c_void| -> T) -> T
    where K: BytesContainer + Eq + Hash, V: BytesContainer
{
    // On Windows we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    match env {
        Some(env) => {
            let mut blk = Vec::new();

            for pair in env.iter() {
                let kv = format!("{}={}",
                                 pair.ref0().container_as_str().unwrap(),
                                 pair.ref1().container_as_str().unwrap());
                blk.extend(kv.utf16_units());
                blk.push(0);
            }

            blk.push(0);

            cb(blk.as_mut_ptr() as *mut c_void)
        }
        _ => cb(ptr::null_mut())
    }
}

fn with_dirp<T>(d: Option<&CString>, cb: |*const u16| -> T) -> T {
    match d {
      Some(dir) => {
          let dir_str = dir.as_str()
                           .expect("expected workingdirectory to be utf-8 encoded");
          let mut dir_str: Vec<u16> = dir_str.utf16_units().collect();
          dir_str.push(0);
          cb(dir_str.as_ptr())
      },
      None => cb(ptr::null())
    }
}

fn free_handle(handle: *mut ()) {
    assert!(unsafe {
        libc::CloseHandle(mem::transmute(handle)) != 0
    })
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_make_command_line() {
        use prelude::*;
        use str;
        use c_str::CString;
        use super::make_command_line;

        fn test_wrapper(prog: &str, args: &[&str]) -> String {
            make_command_line(&prog.to_c_str(),
                              args.iter()
                                  .map(|a| a.to_c_str())
                                  .collect::<Vec<CString>>()
                                  .as_slice())
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
            test_wrapper("\u03c0\u042f\u97f3\u00e6\u221e", &[]),
            "\u03c0\u042f\u97f3\u00e6\u221e"
        );
    }
}
