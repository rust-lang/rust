// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cast;
use libc::{pid_t, c_void, c_int};
use libc;
use os;
use prelude::*;
use ptr;
use rt::io;
use super::file;

/**
 * A value representing a child process.
 *
 * The lifetime of this value is linked to the lifetime of the actual
 * process - the Process destructor calls self.finish() which waits
 * for the process to terminate.
 */
pub struct Process {
    /// The unique id of the process (this should never be negative).
    priv pid: pid_t,

    /// A handle to the process - on unix this will always be NULL, but on
    /// windows it will be a HANDLE to the process, which will prevent the
    /// pid being re-used until the handle is closed.
    priv handle: *(),

    /// Currently known stdin of the child, if any
    priv input: Option<file::FileDesc>,
    /// Currently known stdout of the child, if any
    priv output: Option<file::FileDesc>,
    /// Currently known stderr of the child, if any
    priv error: Option<file::FileDesc>,

    /// None until finish() is called.
    priv exit_code: Option<int>,
}

impl Process {
    /// Creates a new process using native process-spawning abilities provided
    /// by the OS. Operations on this process will be blocking instead of using
    /// the runtime for sleeping just this current task.
    ///
    /// # Arguments
    ///
    /// * prog - the program to run
    /// * args - the arguments to pass to the program, not including the program
    ///          itself
    /// * env - an optional envrionment to specify for the child process. If
    ///         this value is `None`, then the child will inherit the parent's
    ///         environment
    /// * cwd - an optionally specified current working directory of the child,
    ///         defaulting to the parent's current working directory
    /// * stdin, stdout, stderr - These optionally specified file descriptors
    ///     dictate where the stdin/out/err of the child process will go. If
    ///     these are `None`, then this module will bind the input/output to an
    ///     os pipe instead. This process takes ownership of these file
    ///     descriptors, closing them upon destruction of the process.
    pub fn new(prog: &str, args: &[~str], env: Option<~[(~str, ~str)]>,
               cwd: Option<&Path>,
               stdin: Option<file::fd_t>,
               stdout: Option<file::fd_t>,
               stderr: Option<file::fd_t>) -> Process {
        #[fixed_stack_segment]; #[inline(never)];

        let (in_pipe, in_fd) = match stdin {
            None => {
                let pipe = os::pipe();
                (Some(pipe), pipe.input)
            },
            Some(fd) => (None, fd)
        };
        let (out_pipe, out_fd) = match stdout {
            None => {
                let pipe = os::pipe();
                (Some(pipe), pipe.out)
            },
            Some(fd) => (None, fd)
        };
        let (err_pipe, err_fd) = match stderr {
            None => {
                let pipe = os::pipe();
                (Some(pipe), pipe.out)
            },
            Some(fd) => (None, fd)
        };

        let res = spawn_process_os(prog, args, env, cwd,
                                   in_fd, out_fd, err_fd);

        unsafe {
            for pipe in in_pipe.iter() { libc::close(pipe.input); }
            for pipe in out_pipe.iter() { libc::close(pipe.out); }
            for pipe in err_pipe.iter() { libc::close(pipe.out); }
        }

        Process {
            pid: res.pid,
            handle: res.handle,
            input: in_pipe.map(|pipe| file::FileDesc::new(pipe.out)),
            output: out_pipe.map(|pipe| file::FileDesc::new(pipe.input)),
            error: err_pipe.map(|pipe| file::FileDesc::new(pipe.input)),
            exit_code: None,
        }
    }

    /// Returns the unique id of the process
    pub fn id(&self) -> pid_t { self.pid }

    /**
     * Returns an io::Writer that can be used to write to this Process's stdin.
     *
     * Fails if there is no stdinavailable (it's already been removed by
     * take_input)
     */
    pub fn input<'a>(&'a mut self) -> &'a mut io::Writer {
        match self.input {
            Some(ref mut fd) => fd as &mut io::Writer,
            None => fail!("This process has no stdin")
        }
    }

    /**
     * Returns an io::Reader that can be used to read from this Process's
     * stdout.
     *
     * Fails if there is no stdin available (it's already been removed by
     * take_output)
     */
    pub fn output<'a>(&'a mut self) -> &'a mut io::Reader {
        match self.input {
            Some(ref mut fd) => fd as &mut io::Reader,
            None => fail!("This process has no stdout")
        }
    }

    /**
     * Returns an io::Reader that can be used to read from this Process's
     * stderr.
     *
     * Fails if there is no stdin available (it's already been removed by
     * take_error)
     */
    pub fn error<'a>(&'a mut self) -> &'a mut io::Reader {
        match self.error {
            Some(ref mut fd) => fd as &mut io::Reader,
            None => fail!("This process has no stderr")
        }
    }

    /**
     * Takes the stdin of this process, transferring ownership to the caller.
     * Note that when the return value is destroyed, the handle will be closed
     * for the child process.
     */
    pub fn take_input(&mut self) -> Option<~io::Writer> {
        self.input.take().map(|fd| ~fd as ~io::Writer)
    }

    /**
     * Takes the stdout of this process, transferring ownership to the caller.
     * Note that when the return value is destroyed, the handle will be closed
     * for the child process.
     */
    pub fn take_output(&mut self) -> Option<~io::Reader> {
        self.output.take().map(|fd| ~fd as ~io::Reader)
    }

    /**
     * Takes the stderr of this process, transferring ownership to the caller.
     * Note that when the return value is destroyed, the handle will be closed
     * for the child process.
     */
    pub fn take_error(&mut self) -> Option<~io::Reader> {
        self.error.take().map(|fd| ~fd as ~io::Reader)
    }

    pub fn wait(&mut self) -> int {
        for &code in self.exit_code.iter() {
            return code;
        }
        let code = waitpid(self.pid);
        self.exit_code = Some(code);
        return code;
    }

    pub fn signal(&mut self, signum: int) -> Result<(), io::IoError> {
        // if the process has finished, and therefore had waitpid called,
        // and we kill it, then on unix we might ending up killing a
        // newer process that happens to have the same (re-used) id
        match self.exit_code {
            Some(*) => return Err(io::IoError {
                kind: io::OtherIoError,
                desc: "can't kill an exited process",
                detail: None,
            }),
            None => {}
        }
        return unsafe { killpid(self.pid, signum) };

        #[cfg(windows)]
        unsafe fn killpid(pid: pid_t, signal: int) -> Result<(), io::IoError> {
            #[fixed_stack_segment]; #[inline(never)];
            match signal {
                io::process::PleaseExitSignal |
                io::process::MustDieSignal => {
                    libc::funcs::extra::kernel32::TerminateProcess(
                        cast::transmute(pid), 1);
                    Ok(())
                }
                _ => Err(io::IoError {
                    kind: io::OtherIoError,
                    desc: "unsupported signal on windows",
                    detail: None,
                })
            }
        }

        #[cfg(not(windows))]
        unsafe fn killpid(pid: pid_t, signal: int) -> Result<(), io::IoError> {
            #[fixed_stack_segment]; #[inline(never)];
            libc::funcs::posix88::signal::kill(pid, signal as c_int);
            Ok(())
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // close all these handles
        self.take_input();
        self.take_output();
        self.take_error();
        self.wait();
        free_handle(self.handle);
    }
}

struct SpawnProcessResult {
    pid: pid_t,
    handle: *(),
}

#[cfg(windows)]
fn spawn_process_os(prog: &str, args: &[~str],
                    env: Option<~[(~str, ~str)]>,
                    dir: Option<&Path>,
                    in_fd: c_int, out_fd: c_int, err_fd: c_int) -> SpawnProcessResult {
    #[fixed_stack_segment]; #[inline(never)];

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
        CreateProcessA
    };
    use libc::funcs::extra::msvcrt::get_osfhandle;

    use mem;

    unsafe {

        let mut si = zeroed_startupinfo();
        si.cb = mem::size_of::<STARTUPINFO>() as DWORD;
        si.dwFlags = STARTF_USESTDHANDLES;

        let cur_proc = GetCurrentProcess();

        let orig_std_in = get_osfhandle(in_fd) as HANDLE;
        if orig_std_in == INVALID_HANDLE_VALUE as HANDLE {
            fail!("failure in get_osfhandle: {}", os::last_os_error());
        }
        if DuplicateHandle(cur_proc, orig_std_in, cur_proc, &mut si.hStdInput,
                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
            fail!("failure in DuplicateHandle: {}", os::last_os_error());
        }

        let orig_std_out = get_osfhandle(out_fd) as HANDLE;
        if orig_std_out == INVALID_HANDLE_VALUE as HANDLE {
            fail!("failure in get_osfhandle: {}", os::last_os_error());
        }
        if DuplicateHandle(cur_proc, orig_std_out, cur_proc, &mut si.hStdOutput,
                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
            fail!("failure in DuplicateHandle: {}", os::last_os_error());
        }

        let orig_std_err = get_osfhandle(err_fd) as HANDLE;
        if orig_std_err == INVALID_HANDLE_VALUE as HANDLE {
            fail!("failure in get_osfhandle: {}", os::last_os_error());
        }
        if DuplicateHandle(cur_proc, orig_std_err, cur_proc, &mut si.hStdError,
                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
            fail!("failure in DuplicateHandle: {}", os::last_os_error());
        }

        let cmd = make_command_line(prog, args);
        let mut pi = zeroed_process_information();
        let mut create_err = None;

        do with_envp(env) |envp| {
            do with_dirp(dir) |dirp| {
                do cmd.with_c_str |cmdp| {
                    let created = CreateProcessA(ptr::null(), cast::transmute(cmdp),
                                                 ptr::mut_null(), ptr::mut_null(), TRUE,
                                                 0, envp, dirp, &mut si, &mut pi);
                    if created == FALSE {
                        create_err = Some(os::last_os_error());
                    }
                }
            }
        }

        CloseHandle(si.hStdInput);
        CloseHandle(si.hStdOutput);
        CloseHandle(si.hStdError);

        for msg in create_err.iter() {
            fail!("failure in CreateProcess: {}", *msg);
        }

        // We close the thread handle because we don't care about keeping the
        // thread id valid, and we aren't keeping the thread handle around to be
        // able to close it later. We don't close the process handle however
        // because we want the process id to stay valid at least until the
        // calling code closes the process handle.
        CloseHandle(pi.hThread);

        SpawnProcessResult {
            pid: pi.dwProcessId as pid_t,
            handle: pi.hProcess as *()
        }
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
        hStdInput: ptr::mut_null(),
        hStdOutput: ptr::mut_null(),
        hStdError: ptr::mut_null()
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

// FIXME: this is only pub so it can be tested (see issue #4536)
#[cfg(windows)]
pub fn make_command_line(prog: &str, args: &[~str]) -> ~str {
    let mut cmd = ~"";
    append_arg(&mut cmd, prog);
    for arg in args.iter() {
        cmd.push_char(' ');
        append_arg(&mut cmd, *arg);
    }
    return cmd;

    fn append_arg(cmd: &mut ~str, arg: &str) {
        let quote = arg.iter().any(|c| c == ' ' || c == '\t');
        if quote {
            cmd.push_char('"');
        }
        for i in range(0u, arg.len()) {
            append_char_at(cmd, arg, i);
        }
        if quote {
            cmd.push_char('"');
        }
    }

    fn append_char_at(cmd: &mut ~str, arg: &str, i: uint) {
        match arg[i] as char {
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

    fn backslash_run_ends_in_quote(s: &str, mut i: uint) -> bool {
        while i < s.len() && s[i] as char == '\\' {
            i += 1;
        }
        return i < s.len() && s[i] as char == '"';
    }
}

#[cfg(unix)]
fn spawn_process_os(prog: &str, args: &[~str],
                    env: Option<~[(~str, ~str)]>,
                    dir: Option<&Path>,
                    in_fd: c_int, out_fd: c_int, err_fd: c_int) -> SpawnProcessResult {
    #[fixed_stack_segment]; #[inline(never)];

    use libc::funcs::posix88::unistd::{fork, dup2, close, chdir, execvp};
    use libc::funcs::bsd44::getdtablesize;

    mod rustrt {
        #[abi = "cdecl"]
        extern {
            pub fn rust_unset_sigprocmask();
        }
    }

    #[cfg(windows)]
    unsafe fn set_environ(_envp: *c_void) {}
    #[cfg(target_os = "macos")]
    unsafe fn set_environ(envp: *c_void) {
        externfn!(fn _NSGetEnviron() -> *mut *c_void);

        *_NSGetEnviron() = envp;
    }
    #[cfg(not(target_os = "macos"), not(windows))]
    unsafe fn set_environ(envp: *c_void) {
        extern {
            static mut environ: *c_void;
        }
        environ = envp;
    }

    unsafe {

        let pid = fork();
        if pid < 0 {
            fail!("failure in fork: {}", os::last_os_error());
        } else if pid > 0 {
            return SpawnProcessResult {pid: pid, handle: ptr::null()};
        }

        rustrt::rust_unset_sigprocmask();

        if dup2(in_fd, 0) == -1 {
            fail!("failure in dup2(in_fd, 0): {}", os::last_os_error());
        }
        if dup2(out_fd, 1) == -1 {
            fail!("failure in dup2(out_fd, 1): {}", os::last_os_error());
        }
        if dup2(err_fd, 2) == -1 {
            fail!("failure in dup3(err_fd, 2): {}", os::last_os_error());
        }
        // close all other fds
        for fd in range(3, getdtablesize()).invert() {
            close(fd as c_int);
        }

        do with_dirp(dir) |dirp| {
            if !dirp.is_null() && chdir(dirp) == -1 {
                fail!("failure in chdir: {}", os::last_os_error());
            }
        }

        do with_envp(env) |envp| {
            if !envp.is_null() {
                set_environ(envp);
            }
            do with_argv(prog, args) |argv| {
                execvp(*argv, argv);
                // execvp only returns if an error occurred
                fail!("failure in execvp: {}", os::last_os_error());
            }
        }
    }
}

#[cfg(unix)]
fn with_argv<T>(prog: &str, args: &[~str], cb: &fn(**libc::c_char) -> T) -> T {
    use vec;

    // We can't directly convert `str`s into `*char`s, as someone needs to hold
    // a reference to the intermediary byte buffers. So first build an array to
    // hold all the ~[u8] byte strings.
    let mut tmps = vec::with_capacity(args.len() + 1);

    tmps.push(prog.to_c_str());

    for arg in args.iter() {
        tmps.push(arg.to_c_str());
    }

    // Next, convert each of the byte strings into a pointer. This is
    // technically unsafe as the caller could leak these pointers out of our
    // scope.
    let mut ptrs = do tmps.map |tmp| {
        tmp.with_ref(|buf| buf)
    };

    // Finally, make sure we add a null pointer.
    ptrs.push(ptr::null());

    ptrs.as_imm_buf(|buf, _| cb(buf))
}

#[cfg(unix)]
fn with_envp<T>(env: Option<~[(~str, ~str)]>, cb: &fn(*c_void) -> T) -> T {
    use vec;

    // On posixy systems we can pass a char** for envp, which is a
    // null-terminated array of "k=v\n" strings. Like `with_argv`, we have to
    // have a temporary buffer to hold the intermediary `~[u8]` byte strings.
    match env {
        Some(env) => {
            let mut tmps = vec::with_capacity(env.len());

            for pair in env.iter() {
                let kv = format!("{}={}", pair.first(), pair.second());
                tmps.push(kv.to_c_str());
            }

            // Once again, this is unsafe.
            let mut ptrs = do tmps.map |tmp| {
                tmp.with_ref(|buf| buf)
            };
            ptrs.push(ptr::null());

            do ptrs.as_imm_buf |buf, _| {
                unsafe { cb(cast::transmute(buf)) }
            }
        }
        _ => cb(ptr::null())
    }
}

#[cfg(windows)]
fn with_envp<T>(env: Option<~[(~str, ~str)]>, cb: &fn(*mut c_void) -> T) -> T {
    // On win32 we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    match env {
        Some(env) => {
            let mut blk = ~[];

            for pair in env.iter() {
                let kv = format!("{}={}", pair.first(), pair.second());
                blk.push_all(kv.as_bytes());
                blk.push(0);
            }

            blk.push(0);

            do blk.as_imm_buf |p, _len| {
                unsafe { cb(cast::transmute(p)) }
            }
        }
        _ => cb(ptr::mut_null())
    }
}

fn with_dirp<T>(d: Option<&Path>, cb: &fn(*libc::c_char) -> T) -> T {
    match d {
      Some(dir) => dir.with_c_str(|buf| cb(buf)),
      None => cb(ptr::null())
    }
}

#[cfg(windows)]
fn free_handle(handle: *()) {
    #[fixed_stack_segment]; #[inline(never)];
    unsafe {
        libc::funcs::extra::kernel32::CloseHandle(cast::transmute(handle));
    }
}

#[cfg(unix)]
fn free_handle(_handle: *()) {
    // unix has no process handle object, just a pid
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
fn waitpid(pid: pid_t) -> int {
    return waitpid_os(pid);

    #[cfg(windows)]
    fn waitpid_os(pid: pid_t) -> int {
        #[fixed_stack_segment]; #[inline(never)];

        use libc::types::os::arch::extra::DWORD;
        use libc::consts::os::extra::{
            SYNCHRONIZE,
            PROCESS_QUERY_INFORMATION,
            FALSE,
            STILL_ACTIVE,
            INFINITE,
            WAIT_FAILED
        };
        use libc::funcs::extra::kernel32::{
            OpenProcess,
            GetExitCodeProcess,
            CloseHandle,
            WaitForSingleObject
        };

        unsafe {

            let process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_INFORMATION,
                                      FALSE,
                                      pid as DWORD);
            if process.is_null() {
                fail!("failure in OpenProcess: {}", os::last_os_error());
            }

            loop {
                let mut status = 0;
                if GetExitCodeProcess(process, &mut status) == FALSE {
                    CloseHandle(process);
                    fail!("failure in GetExitCodeProcess: {}", os::last_os_error());
                }
                if status != STILL_ACTIVE {
                    CloseHandle(process);
                    return status as int;
                }
                if WaitForSingleObject(process, INFINITE) == WAIT_FAILED {
                    CloseHandle(process);
                    fail!("failure in WaitForSingleObject: {}", os::last_os_error());
                }
            }
        }
    }

    #[cfg(unix)]
    fn waitpid_os(pid: pid_t) -> int {
        #[fixed_stack_segment]; #[inline(never)];

        use libc::funcs::posix01::wait::*;

        #[cfg(target_os = "linux")]
        #[cfg(target_os = "android")]
        fn WIFEXITED(status: i32) -> bool {
            (status & 0xffi32) == 0i32
        }

        #[cfg(target_os = "macos")]
        #[cfg(target_os = "freebsd")]
        fn WIFEXITED(status: i32) -> bool {
            (status & 0x7fi32) == 0i32
        }

        #[cfg(target_os = "linux")]
        #[cfg(target_os = "android")]
        fn WEXITSTATUS(status: i32) -> i32 {
            (status >> 8i32) & 0xffi32
        }

        #[cfg(target_os = "macos")]
        #[cfg(target_os = "freebsd")]
        fn WEXITSTATUS(status: i32) -> i32 {
            status >> 8i32
        }

        let mut status = 0 as c_int;
        if unsafe { waitpid(pid, &mut status, 0) } == -1 {
            fail!("failure in waitpid: {}", os::last_os_error());
        }

        return if WIFEXITED(status) {
            WEXITSTATUS(status) as int
        } else {
            1
        };
    }
}

#[cfg(test)]
mod tests {

    #[test] #[cfg(windows)]
    fn test_make_command_line() {
        use super::make_command_line;
        assert_eq!(
            make_command_line("prog", [~"aaa", ~"bbb", ~"ccc"]),
            ~"prog aaa bbb ccc"
        );
        assert_eq!(
            make_command_line("C:\\Program Files\\blah\\blah.exe", [~"aaa"]),
            ~"\"C:\\Program Files\\blah\\blah.exe\" aaa"
        );
        assert_eq!(
            make_command_line("C:\\Program Files\\test", [~"aa\"bb"]),
            ~"\"C:\\Program Files\\test\" aa\\\"bb"
        );
        assert_eq!(
            make_command_line("echo", [~"a b c"]),
            ~"echo \"a b c\""
        );
    }

    // Currently most of the tests of this functionality live inside std::run,
    // but they may move here eventually as a non-blocking backend is added to
    // std::run
}
