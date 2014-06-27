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
use std::c_str::CString;
use std::io;
use std::mem;
use std::os;
use std::ptr;
use std::rt::rtio::{ProcessConfig, IoResult, IoError};
use std::rt::rtio;

use super::file;
use super::util;

#[cfg(windows)] use std::string::String;
#[cfg(unix)] use super::c;
#[cfg(unix)] use super::retry;
#[cfg(unix)] use io::helper_thread::Helper;

#[cfg(unix)]
helper_init!(static mut HELPER: Helper<Req>)

/**
 * A value representing a child process.
 *
 * The lifetime of this value is linked to the lifetime of the actual
 * process - the Process destructor calls self.finish() which waits
 * for the process to terminate.
 */
pub struct Process {
    /// The unique id of the process (this should never be negative).
    pid: pid_t,

    /// A handle to the process - on unix this will always be NULL, but on
    /// windows it will be a HANDLE to the process, which will prevent the
    /// pid being re-used until the handle is closed.
    handle: *mut (),

    /// None until finish() is called.
    exit_code: Option<rtio::ProcessExit>,

    /// Manually delivered signal
    exit_signal: Option<int>,

    /// Deadline after which wait() will return
    deadline: u64,
}

#[cfg(unix)]
enum Req {
    NewChild(libc::pid_t, Sender<rtio::ProcessExit>, u64),
}

impl Process {
    /// Creates a new process using native process-spawning abilities provided
    /// by the OS. Operations on this process will be blocking instead of using
    /// the runtime for sleeping just this current task.
    pub fn spawn(cfg: ProcessConfig)
        -> IoResult<(Process, Vec<Option<file::FileDesc>>)>
    {
        // right now we only handle stdin/stdout/stderr.
        if cfg.extra_io.len() > 0 {
            return Err(super::unimpl());
        }

        fn get_io(io: rtio::StdioContainer,
                  ret: &mut Vec<Option<file::FileDesc>>)
            -> IoResult<Option<file::FileDesc>>
        {
            match io {
                rtio::Ignored => { ret.push(None); Ok(None) }
                rtio::InheritFd(fd) => {
                    ret.push(None);
                    Ok(Some(file::FileDesc::new(fd, true)))
                }
                rtio::CreatePipe(readable, _writable) => {
                    let (reader, writer) = try!(pipe());
                    let (theirs, ours) = if readable {
                        (reader, writer)
                    } else {
                        (writer, reader)
                    };
                    ret.push(Some(ours));
                    Ok(Some(theirs))
                }
            }
        }

        let mut ret_io = Vec::new();
        let res = spawn_process_os(cfg,
                                   try!(get_io(cfg.stdin, &mut ret_io)),
                                   try!(get_io(cfg.stdout, &mut ret_io)),
                                   try!(get_io(cfg.stderr, &mut ret_io)));

        match res {
            Ok(res) => {
                let p = Process {
                    pid: res.pid,
                    handle: res.handle,
                    exit_code: None,
                    exit_signal: None,
                    deadline: 0,
                };
                Ok((p, ret_io))
            }
            Err(e) => Err(e)
        }
    }

    pub fn kill(pid: libc::pid_t, signum: int) -> IoResult<()> {
        unsafe { killpid(pid, signum) }
    }
}

impl rtio::RtioProcess for Process {
    fn id(&self) -> pid_t { self.pid }

    fn set_timeout(&mut self, timeout: Option<u64>) {
        self.deadline = timeout.map(|i| i + ::io::timer::now()).unwrap_or(0);
    }

    fn wait(&mut self) -> IoResult<rtio::ProcessExit> {
        match self.exit_code {
            Some(code) => Ok(code),
            None => {
                let code = try!(waitpid(self.pid, self.deadline));
                // On windows, waitpid will never return a signal. If a signal
                // was successfully delivered to the process, however, we can
                // consider it as having died via a signal.
                let code = match self.exit_signal {
                    None => code,
                    Some(signal) if cfg!(windows) => rtio::ExitSignal(signal),
                    Some(..) => code,
                };
                self.exit_code = Some(code);
                Ok(code)
            }
        }
    }

    fn kill(&mut self, signum: int) -> IoResult<()> {
        #[cfg(unix)] use ERROR = libc::EINVAL;
        #[cfg(windows)] use ERROR = libc::ERROR_NOTHING_TO_TERMINATE;

        // On linux (and possibly other unices), a process that has exited will
        // continue to accept signals because it is "defunct". The delivery of
        // signals will only fail once the child has been reaped. For this
        // reason, if the process hasn't exited yet, then we attempt to collect
        // their status with WNOHANG.
        if self.exit_code.is_none() {
            match waitpid_nowait(self.pid) {
                Some(code) => { self.exit_code = Some(code); }
                None => {}
            }
        }

        // if the process has finished, and therefore had waitpid called,
        // and we kill it, then on unix we might ending up killing a
        // newer process that happens to have the same (re-used) id
        match self.exit_code {
            Some(..) => return Err(IoError {
                code: ERROR as uint,
                extra: 0,
                detail: Some("can't kill an exited process".to_str()),
            }),
            None => {}
        }

        // A successfully delivered signal that isn't 0 (just a poll for being
        // alive) is recorded for windows (see wait())
        match unsafe { killpid(self.pid, signum) } {
            Ok(()) if signum == 0 => Ok(()),
            Ok(()) => { self.exit_signal = Some(signum); Ok(()) }
            Err(e) => Err(e),
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        free_handle(self.handle);
    }
}

fn pipe() -> IoResult<(file::FileDesc, file::FileDesc)> {
    #[cfg(unix)] use ERROR = libc::EMFILE;
    #[cfg(windows)] use ERROR = libc::WSAEMFILE;
    struct Closer { fd: libc::c_int }

    let os::Pipe { reader, writer } = match unsafe { os::pipe() } {
        Ok(p) => p,
        Err(io::IoError { detail, .. }) => return Err(IoError {
            code: ERROR as uint,
            extra: 0,
            detail: detail,
        })
    };
    let mut reader = Closer { fd: reader };
    let mut writer = Closer { fd: writer };

    let native_reader = file::FileDesc::new(reader.fd, true);
    reader.fd = -1;
    let native_writer = file::FileDesc::new(writer.fd, true);
    writer.fd = -1;
    return Ok((native_reader, native_writer));

    impl Drop for Closer {
        fn drop(&mut self) {
            if self.fd != -1 {
                let _ = unsafe { libc::close(self.fd) };
            }
        }
    }
}

#[cfg(windows)]
unsafe fn killpid(pid: pid_t, signal: int) -> IoResult<()> {
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
                    code: libc::ERROR_NOTHING_TO_TERMINATE as uint,
                    extra: 0,
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
            code: libc::ERROR_CALL_NOT_IMPLEMENTED as uint,
            extra: 0,
            detail: Some("unsupported signal on windows".to_string()),
        })
    };
    let _ = libc::CloseHandle(handle);
    return ret;
}

#[cfg(not(windows))]
unsafe fn killpid(pid: pid_t, signal: int) -> IoResult<()> {
    let r = libc::funcs::posix88::signal::kill(pid, signal as c_int);
    super::mkerr_libc(r)
}

struct SpawnProcessResult {
    pid: pid_t,
    handle: *mut (),
}

#[cfg(windows)]
fn spawn_process_os(cfg: ProcessConfig,
                    in_fd: Option<file::FileDesc>,
                    out_fd: Option<file::FileDesc>,
                    err_fd: Option<file::FileDesc>)
                 -> IoResult<SpawnProcessResult> {
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

    use std::mem;

    if cfg.gid.is_some() || cfg.uid.is_some() {
        return Err(IoError {
            code: libc::ERROR_CALL_NOT_IMPLEMENTED as uint,
            extra: 0,
            detail: Some("unsupported gid/uid requested on windows".to_str()),
        })
    }

    unsafe {
        let mut si = zeroed_startupinfo();
        si.cb = mem::size_of::<STARTUPINFO>() as DWORD;
        si.dwFlags = STARTF_USESTDHANDLES;

        let cur_proc = GetCurrentProcess();

        // Similarly to unix, we don't actually leave holes for the stdio file
        // descriptors, but rather open up /dev/null equivalents. These
        // equivalents are drawn from libuv's windows process spawning.
        let set_fd = |fd: &Option<file::FileDesc>, slot: &mut HANDLE,
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
                        lpSecurityDescriptor: ptr::mut_null(),
                        bInheritHandle: 1,
                    };
                    let filename = "NUL".to_utf16().append_one(0);
                    *slot = libc::CreateFileW(filename.as_ptr(),
                                              access,
                                              libc::FILE_SHARE_READ |
                                                  libc::FILE_SHARE_WRITE,
                                              &mut sa,
                                              libc::OPEN_EXISTING,
                                              0,
                                              ptr::mut_null());
                    if *slot == INVALID_HANDLE_VALUE as libc::HANDLE {
                        return Err(super::last_error())
                    }
                }
                Some(ref fd) => {
                    let orig = get_osfhandle(fd.fd()) as HANDLE;
                    if orig == INVALID_HANDLE_VALUE as HANDLE {
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

        let cmd_str = make_command_line(cfg.program, cfg.args);
        let mut pi = zeroed_process_information();
        let mut create_err = None;

        // stolen from the libuv code.
        let mut flags = libc::CREATE_UNICODE_ENVIRONMENT;
        if cfg.detach {
            flags |= libc::DETACHED_PROCESS | libc::CREATE_NEW_PROCESS_GROUP;
        }

        with_envp(cfg.env, |envp| {
            with_dirp(cfg.cwd, |dirp| {
                let mut cmd_str = cmd_str.to_utf16().append_one(0);
                let created = CreateProcessW(ptr::null(),
                                             cmd_str.as_mut_ptr(),
                                             ptr::mut_null(),
                                             ptr::mut_null(),
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

        Ok(SpawnProcessResult {
            pid: pi.dwProcessId as pid_t,
            handle: pi.hProcess as *mut ()
        })
    }
}

#[cfg(windows)]
fn zeroed_startupinfo() -> libc::types::os::arch::extra::STARTUPINFO {
    libc::types::os::arch::extra::STARTUPINFO {
        cb: 0,
        lpReserved: ptr::mut_null(),
        lpDesktop: ptr::mut_null(),
        lpTitle: ptr::mut_null(),
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
        lpReserved2: ptr::mut_null(),
        hStdInput: libc::INVALID_HANDLE_VALUE as libc::HANDLE,
        hStdOutput: libc::INVALID_HANDLE_VALUE as libc::HANDLE,
        hStdError: libc::INVALID_HANDLE_VALUE as libc::HANDLE,
    }
}

#[cfg(windows)]
fn zeroed_process_information() -> libc::types::os::arch::extra::PROCESS_INFORMATION {
    libc::types::os::arch::extra::PROCESS_INFORMATION {
        hProcess: ptr::mut_null(),
        hThread: ptr::mut_null(),
        dwProcessId: 0,
        dwThreadId: 0
    }
}

#[cfg(windows)]
fn make_command_line(prog: &CString, args: &[CString]) -> String {
    let mut cmd = String::new();
    append_arg(&mut cmd, prog.as_str()
                             .expect("expected program name to be utf-8 encoded"));
    for arg in args.iter() {
        cmd.push_char(' ');
        append_arg(&mut cmd, arg.as_str()
                                .expect("expected argument to be utf-8 encoded"));
    }
    return cmd;

    fn append_arg(cmd: &mut String, arg: &str) {
        let quote = arg.chars().any(|c| c == ' ' || c == '\t');
        if quote {
            cmd.push_char('"');
        }
        let argvec: Vec<char> = arg.chars().collect();
        for i in range(0u, argvec.len()) {
            append_char_at(cmd, &argvec, i);
        }
        if quote {
            cmd.push_char('"');
        }
    }

    fn append_char_at(cmd: &mut String, arg: &Vec<char>, i: uint) {
        match *arg.get(i) {
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
                    cmd.push_char('\\');
                }
            }
            c => {
                cmd.push_char(c);
            }
        }
    }

    fn backslash_run_ends_in_quote(s: &Vec<char>, mut i: uint) -> bool {
        while i < s.len() && *s.get(i) == '\\' {
            i += 1;
        }
        return i < s.len() && *s.get(i) == '"';
    }
}

#[cfg(unix)]
fn spawn_process_os(cfg: ProcessConfig,
                    in_fd: Option<file::FileDesc>,
                    out_fd: Option<file::FileDesc>,
                    err_fd: Option<file::FileDesc>)
                -> IoResult<SpawnProcessResult>
{
    use libc::funcs::posix88::unistd::{fork, dup2, close, chdir, execvp};
    use libc::funcs::bsd44::getdtablesize;
    use io::c;

    mod rustrt {
        extern {
            pub fn rust_unset_sigprocmask();
        }
    }

    #[cfg(target_os = "macos")]
    unsafe fn set_environ(envp: *const c_void) {
        extern { fn _NSGetEnviron() -> *mut *const c_void; }

        *_NSGetEnviron() = envp;
    }
    #[cfg(not(target_os = "macos"))]
    unsafe fn set_environ(envp: *const c_void) {
        extern { static mut environ: *const c_void; }
        environ = envp;
    }

    unsafe fn set_cloexec(fd: c_int) {
        let ret = c::ioctl(fd, c::FIOCLEX);
        assert_eq!(ret, 0);
    }

    let dirp = cfg.cwd.map(|c| c.as_ptr()).unwrap_or(ptr::null());

    let cfg = unsafe {
        mem::transmute::<ProcessConfig,ProcessConfig<'static>>(cfg)
    };

    with_envp(cfg.env, proc(envp) {
        with_argv(cfg.program, cfg.args, proc(argv) unsafe {
            let (mut input, mut output) = try!(pipe());

            // We may use this in the child, so perform allocations before the
            // fork
            let devnull = "/dev/null".to_c_str();

            set_cloexec(output.fd());

            let pid = fork();
            if pid < 0 {
                return Err(super::last_error())
            } else if pid > 0 {
                drop(output);
                let mut bytes = [0, ..4];
                return match input.inner_read(bytes) {
                    Ok(4) => {
                        let errno = (bytes[0] << 24) as i32 |
                                    (bytes[1] << 16) as i32 |
                                    (bytes[2] <<  8) as i32 |
                                    (bytes[3] <<  0) as i32;
                        Err(IoError {
                            code: errno as uint,
                            detail: None,
                            extra: 0,
                        })
                    }
                    Err(..) => {
                        Ok(SpawnProcessResult {
                            pid: pid,
                            handle: ptr::mut_null()
                        })
                    }
                    Ok(..) => fail!("short read on the cloexec pipe"),
                };
            }
            // And at this point we've reached a special time in the life of the
            // child. The child must now be considered hamstrung and unable to
            // do anything other than syscalls really. Consider the following
            // scenario:
            //
            //      1. Thread A of process 1 grabs the malloc() mutex
            //      2. Thread B of process 1 forks(), creating thread C
            //      3. Thread C of process 2 then attempts to malloc()
            //      4. The memory of process 2 is the same as the memory of
            //         process 1, so the mutex is locked.
            //
            // This situation looks a lot like deadlock, right? It turns out
            // that this is what pthread_atfork() takes care of, which is
            // presumably implemented across platforms. The first thing that
            // threads to *before* forking is to do things like grab the malloc
            // mutex, and then after the fork they unlock it.
            //
            // Despite this information, libnative's spawn has been witnessed to
            // deadlock on both OSX and FreeBSD. I'm not entirely sure why, but
            // all collected backtraces point at malloc/free traffic in the
            // child spawned process.
            //
            // For this reason, the block of code below should contain 0
            // invocations of either malloc of free (or their related friends).
            //
            // As an example of not having malloc/free traffic, we don't close
            // this file descriptor by dropping the FileDesc (which contains an
            // allocation). Instead we just close it manually. This will never
            // have the drop glue anyway because this code never returns (the
            // child will either exec() or invoke libc::exit)
            let _ = libc::close(input.fd());

            fn fail(output: &mut file::FileDesc) -> ! {
                let errno = os::errno();
                let bytes = [
                    (errno << 24) as u8,
                    (errno << 16) as u8,
                    (errno <<  8) as u8,
                    (errno <<  0) as u8,
                ];
                assert!(output.inner_write(bytes).is_ok());
                unsafe { libc::_exit(1) }
            }

            rustrt::rust_unset_sigprocmask();

            // If a stdio file descriptor is set to be ignored (via a -1 file
            // descriptor), then we don't actually close it, but rather open
            // up /dev/null into that file descriptor. Otherwise, the first file
            // descriptor opened up in the child would be numbered as one of the
            // stdio file descriptors, which is likely to wreak havoc.
            let setup = |src: Option<file::FileDesc>, dst: c_int| {
                let src = match src {
                    None => {
                        let flags = if dst == libc::STDIN_FILENO {
                            libc::O_RDONLY
                        } else {
                            libc::O_RDWR
                        };
                        libc::open(devnull.as_ptr(), flags, 0)
                    }
                    Some(obj) => {
                        let fd = obj.fd();
                        // Leak the memory and the file descriptor. We're in the
                        // child now an all our resources are going to be
                        // cleaned up very soon
                        mem::forget(obj);
                        fd
                    }
                };
                src != -1 && retry(|| dup2(src, dst)) != -1
            };

            if !setup(in_fd, libc::STDIN_FILENO) { fail(&mut output) }
            if !setup(out_fd, libc::STDOUT_FILENO) { fail(&mut output) }
            if !setup(err_fd, libc::STDERR_FILENO) { fail(&mut output) }

            // close all other fds
            for fd in range(3, getdtablesize()).rev() {
                if fd != output.fd() {
                    let _ = close(fd as c_int);
                }
            }

            match cfg.gid {
                Some(u) => {
                    if libc::setgid(u as libc::gid_t) != 0 {
                        fail(&mut output);
                    }
                }
                None => {}
            }
            match cfg.uid {
                Some(u) => {
                    // When dropping privileges from root, the `setgroups` call
                    // will remove any extraneous groups. If we don't call this,
                    // then even though our uid has dropped, we may still have
                    // groups that enable us to do super-user things. This will
                    // fail if we aren't root, so don't bother checking the
                    // return value, this is just done as an optimistic
                    // privilege dropping function.
                    extern {
                        fn setgroups(ngroups: libc::c_int,
                                     ptr: *const libc::c_void) -> libc::c_int;
                    }
                    let _ = setgroups(0, 0 as *const libc::c_void);

                    if libc::setuid(u as libc::uid_t) != 0 {
                        fail(&mut output);
                    }
                }
                None => {}
            }
            if cfg.detach {
                // Don't check the error of setsid because it fails if we're the
                // process leader already. We just forked so it shouldn't return
                // error, but ignore it anyway.
                let _ = libc::setsid();
            }
            if !dirp.is_null() && chdir(dirp) == -1 {
                fail(&mut output);
            }
            if !envp.is_null() {
                set_environ(envp);
            }
            let _ = execvp(*argv, argv as *mut _);
            fail(&mut output);
        })
    })
}

#[cfg(unix)]
fn with_argv<T>(prog: &CString, args: &[CString],
                cb: proc(*const *const libc::c_char) -> T) -> T {
    let mut ptrs: Vec<*const libc::c_char> = Vec::with_capacity(args.len()+1);

    // Convert the CStrings into an array of pointers. Note: the
    // lifetime of the various CStrings involved is guaranteed to be
    // larger than the lifetime of our invocation of cb, but this is
    // technically unsafe as the callback could leak these pointers
    // out of our scope.
    ptrs.push(prog.as_ptr());
    ptrs.extend(args.iter().map(|tmp| tmp.as_ptr()));

    // Add a terminating null pointer (required by libc).
    ptrs.push(ptr::null());

    cb(ptrs.as_ptr())
}

#[cfg(unix)]
fn with_envp<T>(env: Option<&[(CString, CString)]>,
                cb: proc(*const c_void) -> T) -> T {
    // On posixy systems we can pass a char** for envp, which is a
    // null-terminated array of "k=v\0" strings. Since we must create
    // these strings locally, yet expose a raw pointer to them, we
    // create a temporary vector to own the CStrings that outlives the
    // call to cb.
    match env {
        Some(env) => {
            let mut tmps = Vec::with_capacity(env.len());

            for pair in env.iter() {
                let mut kv = Vec::new();
                kv.push_all(pair.ref0().as_bytes_no_nul());
                kv.push('=' as u8);
                kv.push_all(pair.ref1().as_bytes()); // includes terminal \0
                tmps.push(kv);
            }

            // As with `with_argv`, this is unsafe, since cb could leak the pointers.
            let mut ptrs: Vec<*const libc::c_char> =
                tmps.iter()
                    .map(|tmp| tmp.as_ptr() as *const libc::c_char)
                    .collect();
            ptrs.push(ptr::null());

            cb(ptrs.as_ptr() as *const c_void)
        }
        _ => cb(ptr::null())
    }
}

#[cfg(windows)]
fn with_envp<T>(env: Option<&[(CString, CString)]>, cb: |*mut c_void| -> T) -> T {
    // On win32 we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    match env {
        Some(env) => {
            let mut blk = Vec::new();

            for pair in env.iter() {
                let kv = format!("{}={}",
                                 pair.ref0().as_str().unwrap(),
                                 pair.ref1().as_str().unwrap());
                blk.push_all(kv.to_utf16().as_slice());
                blk.push(0);
            }

            blk.push(0);

            cb(blk.as_mut_ptr() as *mut c_void)
        }
        _ => cb(ptr::mut_null())
    }
}

#[cfg(windows)]
fn with_dirp<T>(d: Option<&CString>, cb: |*const u16| -> T) -> T {
    match d {
      Some(dir) => {
          let dir_str = dir.as_str()
                           .expect("expected workingdirectory to be utf-8 encoded");
          let dir_str = dir_str.to_utf16().append_one(0);
          cb(dir_str.as_ptr())
      },
      None => cb(ptr::null())
    }
}

#[cfg(windows)]
fn free_handle(handle: *mut ()) {
    assert!(unsafe {
        libc::CloseHandle(mem::transmute(handle)) != 0
    })
}

#[cfg(unix)]
fn free_handle(_handle: *mut ()) {
    // unix has no process handle object, just a pid
}

#[cfg(unix)]
fn translate_status(status: c_int) -> rtio::ProcessExit {
    #![allow(non_snake_case_functions)]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    mod imp {
        pub fn WIFEXITED(status: i32) -> bool { (status & 0xff) == 0 }
        pub fn WEXITSTATUS(status: i32) -> i32 { (status >> 8) & 0xff }
        pub fn WTERMSIG(status: i32) -> i32 { status & 0x7f }
    }

    #[cfg(target_os = "macos")]
    #[cfg(target_os = "ios")]
    #[cfg(target_os = "freebsd")]
    mod imp {
        pub fn WIFEXITED(status: i32) -> bool { (status & 0x7f) == 0 }
        pub fn WEXITSTATUS(status: i32) -> i32 { status >> 8 }
        pub fn WTERMSIG(status: i32) -> i32 { status & 0o177 }
    }

    if imp::WIFEXITED(status) {
        rtio::ExitStatus(imp::WEXITSTATUS(status) as int)
    } else {
        rtio::ExitSignal(imp::WTERMSIG(status) as int)
    }
}

/**
 * Waits for a process to exit and returns the exit code, failing
 * if there is no process with the specified id.
 *
 * Note that this is private to avoid race conditions on unix where if
 * a user calls waitpid(some_process.get_id()) then some_process.finish()
 * and some_process.destroy() and some_process.finalize() will then either
 * operate on a none-existent process or, even worse, on a newer process
 * with the same id.
 */
#[cfg(windows)]
fn waitpid(pid: pid_t, deadline: u64) -> IoResult<rtio::ProcessExit> {
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
                                  pid as DWORD);
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
                return Ok(rtio::ExitStatus(status as int));
            }
            let interval = if deadline == 0 {
                INFINITE
            } else {
                let now = ::io::timer::now();
                if deadline < now {0} else {(deadline - now) as u32}
            };
            match WaitForSingleObject(process, interval) {
                WAIT_OBJECT_0 => {}
                WAIT_TIMEOUT => {
                    assert!(CloseHandle(process) != 0);
                    return Err(util::timeout("process wait timed out"))
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

#[cfg(unix)]
fn waitpid(pid: pid_t, deadline: u64) -> IoResult<rtio::ProcessExit> {
    use std::cmp;
    use std::comm;

    static mut WRITE_FD: libc::c_int = 0;

    let mut status = 0 as c_int;
    if deadline == 0 {
        return match retry(|| unsafe { c::waitpid(pid, &mut status, 0) }) {
            -1 => fail!("unknown waitpid error: {}", super::last_error().code),
            _ => Ok(translate_status(status)),
        }
    }

    // On unix, wait() and its friends have no timeout parameters, so there is
    // no way to time out a thread in wait(). From some googling and some
    // thinking, it appears that there are a few ways to handle timeouts in
    // wait(), but the only real reasonable one for a multi-threaded program is
    // to listen for SIGCHLD.
    //
    // With this in mind, the waiting mechanism with a timeout barely uses
    // waitpid() at all. There are a few times that waitpid() is invoked with
    // WNOHANG, but otherwise all the necessary blocking is done by waiting for
    // a SIGCHLD to arrive (and that blocking has a timeout). Note, however,
    // that waitpid() is still used to actually reap the child.
    //
    // Signal handling is super tricky in general, and this is no exception. Due
    // to the async nature of SIGCHLD, we use the self-pipe trick to transmit
    // data out of the signal handler to the rest of the application. The first
    // idea would be to have each thread waiting with a timeout to read this
    // output file descriptor, but a write() is akin to a signal(), not a
    // broadcast(), so it would only wake up one thread, and possibly the wrong
    // thread. Hence a helper thread is used.
    //
    // The helper thread here is responsible for farming requests for a
    // waitpid() with a timeout, and then processing all of the wait requests.
    // By guaranteeing that only this helper thread is reading half of the
    // self-pipe, we're sure that we'll never lose a SIGCHLD. This helper thread
    // is also responsible for select() to wait for incoming messages or
    // incoming SIGCHLD messages, along with passing an appropriate timeout to
    // select() to wake things up as necessary.
    //
    // The ordering of the following statements is also very purposeful. First,
    // we must be guaranteed that the helper thread is booted and available to
    // receive SIGCHLD signals, and then we must also ensure that we do a
    // nonblocking waitpid() at least once before we go ask the sigchld helper.
    // This prevents the race where the child exits, we boot the helper, and
    // then we ask for the child's exit status (never seeing a sigchld).
    //
    // The actual communication between the helper thread and this thread is
    // quite simple, just a channel moving data around.

    unsafe { HELPER.boot(register_sigchld, waitpid_helper) }

    match waitpid_nowait(pid) {
        Some(ret) => return Ok(ret),
        None => {}
    }

    let (tx, rx) = channel();
    unsafe { HELPER.send(NewChild(pid, tx, deadline)); }
    return match rx.recv_opt() {
        Ok(e) => Ok(e),
        Err(()) => Err(util::timeout("wait timed out")),
    };

    // Register a new SIGCHLD handler, returning the reading half of the
    // self-pipe plus the old handler registered (return value of sigaction).
    //
    // Be sure to set up the self-pipe first because as soon as we register a
    // handler we're going to start receiving signals.
    fn register_sigchld() -> (libc::c_int, c::sigaction) {
        unsafe {
            let mut pipes = [0, ..2];
            assert_eq!(libc::pipe(pipes.as_mut_ptr()), 0);
            util::set_nonblocking(pipes[0], true).ok().unwrap();
            util::set_nonblocking(pipes[1], true).ok().unwrap();
            WRITE_FD = pipes[1];

            let mut old: c::sigaction = mem::zeroed();
            let mut new: c::sigaction = mem::zeroed();
            new.sa_handler = sigchld_handler;
            new.sa_flags = c::SA_NOCLDSTOP;
            assert_eq!(c::sigaction(c::SIGCHLD, &new, &mut old), 0);
            (pipes[0], old)
        }
    }

    // Helper thread for processing SIGCHLD messages
    fn waitpid_helper(input: libc::c_int,
                      messages: Receiver<Req>,
                      (read_fd, old): (libc::c_int, c::sigaction)) {
        util::set_nonblocking(input, true).ok().unwrap();
        let mut set: c::fd_set = unsafe { mem::zeroed() };
        let mut tv: libc::timeval;
        let mut active = Vec::<(libc::pid_t, Sender<rtio::ProcessExit>, u64)>::new();
        let max = cmp::max(input, read_fd) + 1;

        'outer: loop {
            // Figure out the timeout of our syscall-to-happen. If we're waiting
            // for some processes, then they'll have a timeout, otherwise we
            // wait indefinitely for a message to arrive.
            //
            // FIXME: sure would be nice to not have to scan the entire array
            let min = active.iter().map(|a| *a.ref2()).enumerate().min_by(|p| {
                p.val1()
            });
            let (p, idx) = match min {
                Some((idx, deadline)) => {
                    let now = ::io::timer::now();
                    let ms = if now < deadline {deadline - now} else {0};
                    tv = util::ms_to_timeval(ms);
                    (&mut tv as *mut _, idx)
                }
                None => (ptr::mut_null(), -1),
            };

            // Wait for something to happen
            c::fd_set(&mut set, input);
            c::fd_set(&mut set, read_fd);
            match unsafe { c::select(max, &mut set, ptr::mut_null(),
                                     ptr::mut_null(), p) } {
                // interrupted, retry
                -1 if os::errno() == libc::EINTR as int => continue,

                // We read something, break out and process
                1 | 2 => {}

                // Timeout, the pending request is removed
                0 => {
                    drop(active.remove(idx));
                    continue
                }

                n => fail!("error in select {} ({})", os::errno(), n),
            }

            // Process any pending messages
            if drain(input) {
                loop {
                    match messages.try_recv() {
                        Ok(NewChild(pid, tx, deadline)) => {
                            active.push((pid, tx, deadline));
                        }
                        Err(comm::Disconnected) => {
                            assert!(active.len() == 0);
                            break 'outer;
                        }
                        Err(comm::Empty) => break,
                    }
                }
            }

            // If a child exited (somehow received SIGCHLD), then poll all
            // children to see if any of them exited.
            //
            // We also attempt to be responsible netizens when dealing with
            // SIGCHLD by invoking any previous SIGCHLD handler instead of just
            // ignoring any previous SIGCHLD handler. Note that we don't provide
            // a 1:1 mapping of our handler invocations to the previous handler
            // invocations because we drain the `read_fd` entirely. This is
            // probably OK because the kernel is already allowed to coalesce
            // simultaneous signals, we're just doing some extra coalescing.
            //
            // Another point of note is that this likely runs the signal handler
            // on a different thread than the one that received the signal. I
            // *think* this is ok at this time.
            //
            // The main reason for doing this is to allow stdtest to run native
            // tests as well. Both libgreen and libnative are running around
            // with process timeouts, but libgreen should get there first
            // (currently libuv doesn't handle old signal handlers).
            if drain(read_fd) {
                let i: uint = unsafe { mem::transmute(old.sa_handler) };
                if i != 0 {
                    assert!(old.sa_flags & c::SA_SIGINFO == 0);
                    (old.sa_handler)(c::SIGCHLD);
                }

                // FIXME: sure would be nice to not have to scan the entire
                //        array...
                active.retain(|&(pid, ref tx, _)| {
                    match waitpid_nowait(pid) {
                        Some(msg) => { tx.send(msg); false }
                        None => true,
                    }
                });
            }
        }

        // Once this helper thread is done, we re-register the old sigchld
        // handler and close our intermediate file descriptors.
        unsafe {
            assert_eq!(c::sigaction(c::SIGCHLD, &old, ptr::mut_null()), 0);
            let _ = libc::close(read_fd);
            let _ = libc::close(WRITE_FD);
            WRITE_FD = -1;
        }
    }

    // Drain all pending data from the file descriptor, returning if any data
    // could be drained. This requires that the file descriptor is in
    // nonblocking mode.
    fn drain(fd: libc::c_int) -> bool {
        let mut ret = false;
        loop {
            let mut buf = [0u8, ..1];
            match unsafe {
                libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void,
                           buf.len() as libc::size_t)
            } {
                n if n > 0 => { ret = true; }
                0 => return true,
                -1 if util::wouldblock() => return ret,
                n => fail!("bad read {} ({})", os::last_os_error(), n),
            }
        }
    }

    // Signal handler for SIGCHLD signals, must be async-signal-safe!
    //
    // This function will write to the writing half of the "self pipe" to wake
    // up the helper thread if it's waiting. Note that this write must be
    // nonblocking because if it blocks and the reader is the thread we
    // interrupted, then we'll deadlock.
    //
    // When writing, if the write returns EWOULDBLOCK then we choose to ignore
    // it. At that point we're guaranteed that there's something in the pipe
    // which will wake up the other end at some point, so we just allow this
    // signal to be coalesced with the pending signals on the pipe.
    extern fn sigchld_handler(_signum: libc::c_int) {
        let msg = 1i;
        match unsafe {
            libc::write(WRITE_FD, &msg as *const _ as *const libc::c_void, 1)
        } {
            1 => {}
            -1 if util::wouldblock() => {} // see above comments
            n => fail!("bad error on write fd: {} {}", n, os::errno()),
        }
    }
}

fn waitpid_nowait(pid: pid_t) -> Option<rtio::ProcessExit> {
    return waitpid_os(pid);

    // This code path isn't necessary on windows
    #[cfg(windows)]
    fn waitpid_os(_pid: pid_t) -> Option<rtio::ProcessExit> { None }

    #[cfg(unix)]
    fn waitpid_os(pid: pid_t) -> Option<rtio::ProcessExit> {
        let mut status = 0 as c_int;
        match retry(|| unsafe {
            c::waitpid(pid, &mut status, c::WNOHANG)
        }) {
            n if n == pid => Some(translate_status(status)),
            0 => None,
            n => fail!("unknown waitpid error `{}`: {}", n,
                       super::last_error().code),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test] #[cfg(windows)]
    fn test_make_command_line() {
        use std::str;
        use std::c_str::CString;
        use super::make_command_line;

        fn test_wrapper(prog: &str, args: &[&str]) -> String {
            make_command_line(&prog.to_c_str(),
                              args.iter()
                                  .map(|a| a.to_c_str())
                                  .collect::<Vec<CString>>()
                                  .as_slice())
        }

        assert_eq!(
            test_wrapper("prog", ["aaa", "bbb", "ccc"]),
            "prog aaa bbb ccc".to_string()
        );

        assert_eq!(
            test_wrapper("C:\\Program Files\\blah\\blah.exe", ["aaa"]),
            "\"C:\\Program Files\\blah\\blah.exe\" aaa".to_string()
        );
        assert_eq!(
            test_wrapper("C:\\Program Files\\test", ["aa\"bb"]),
            "\"C:\\Program Files\\test\" aa\\\"bb".to_string()
        );
        assert_eq!(
            test_wrapper("echo", ["a b c"]),
            "echo \"a b c\"".to_string()
        );
        assert_eq!(
            test_wrapper("\u03c0\u042f\u97f3\u00e6\u221e", []),
            "\u03c0\u042f\u97f3\u00e6\u221e".to_string()
        );
    }
}
