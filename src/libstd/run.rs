// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Process spawning.

#[allow(missing_doc)];

use iterator::IteratorUtil;
use cast;
use comm::{stream, SharedChan, GenericChan, GenericPort};
use int;
use io;
use libc::{pid_t, c_void, c_int};
use libc;
use option::{Some, None};
use os;
use prelude::*;
use ptr;
use str;
use task;
use vec;

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

    /**
     * A handle to the process - on unix this will always be NULL, but on
     * windows it will be a HANDLE to the process, which will prevent the
     * pid being re-used until the handle is closed.
     */
    priv handle: *(),

    /// Some(fd), or None when stdin is being redirected from a fd not created by Process::new.
    priv input: Option<c_int>,

    /// Some(file), or None when stdout is being redirected to a fd not created by Process::new.
    priv output: Option<*libc::FILE>,

    /// Some(file), or None when stderr is being redirected to a fd not created by Process::new.
    priv error: Option<*libc::FILE>,

    /// None until finish() is called.
    priv exit_code: Option<int>,
}

/// Options that can be given when starting a Process.
pub struct ProcessOptions<'self> {

    /**
     * If this is None then the new process will have the same initial
     * environment as the parent process.
     *
     * If this is Some(vec-of-names-and-values) then the new process will
     * have an environment containing the given named values only.
     */
    env: Option<&'self [(~str, ~str)]>,

    /**
     * If this is None then the new process will use the same initial working
     * directory as the parent process.
     *
     * If this is Some(path) then the new process will use the given path
     * for its initial working directory.
     */
    dir: Option<&'self Path>,

    /**
     * If this is None then a new pipe will be created for the new process's
     * input and Process.input() will provide a Writer to write to this pipe.
     *
     * If this is Some(file-descriptor) then the new process will read its input
     * from the given file descriptor, Process.input_redirected() will return
     * true, and Process.input() will fail.
     */
    in_fd: Option<c_int>,

    /**
     * If this is None then a new pipe will be created for the new progam's
     * output and Process.output() will provide a Reader to read from this pipe.
     *
     * If this is Some(file-descriptor) then the new process will write its output
     * to the given file descriptor, Process.output_redirected() will return
     * true, and Process.output() will fail.
     */
    out_fd: Option<c_int>,

    /**
     * If this is None then a new pipe will be created for the new progam's
     * error stream and Process.error() will provide a Reader to read from this pipe.
     *
     * If this is Some(file-descriptor) then the new process will write its error output
     * to the given file descriptor, Process.error_redirected() will return true, and
     * and Process.error() will fail.
     */
    err_fd: Option<c_int>,
}

impl <'self> ProcessOptions<'self> {
    /// Return a ProcessOptions that has None in every field.
    pub fn new<'a>() -> ProcessOptions<'a> {
        ProcessOptions {
            env: None,
            dir: None,
            in_fd: None,
            out_fd: None,
            err_fd: None,
        }
    }
}

/// The output of a finished process.
pub struct ProcessOutput {

    /// The status (exit code) of the process.
    status: int,

    /// The data that the process wrote to stdout.
    output: ~[u8],

    /// The data that the process wrote to stderr.
    error: ~[u8],
}

impl Process {
    /**
     * Spawns a new Process.
     *
     * # Arguments
     *
     * * prog - The path to an executable.
     * * args - Vector of arguments to pass to the child process.
     * * options - Options to configure the environment of the process,
     *             the working directory and the standard IO streams.
     */
    pub fn new(prog: &str, args: &[~str], options: ProcessOptions)
               -> Process {
        let (in_pipe, in_fd) = match options.in_fd {
            None => {
                let pipe = os::pipe();
                (Some(pipe), pipe.in)
            },
            Some(fd) => (None, fd)
        };
        let (out_pipe, out_fd) = match options.out_fd {
            None => {
                let pipe = os::pipe();
                (Some(pipe), pipe.out)
            },
            Some(fd) => (None, fd)
        };
        let (err_pipe, err_fd) = match options.err_fd {
            None => {
                let pipe = os::pipe();
                (Some(pipe), pipe.out)
            },
            Some(fd) => (None, fd)
        };

        let res = spawn_process_os(prog, args, options.env, options.dir,
                                   in_fd, out_fd, err_fd);

        unsafe {
            for in_pipe.each  |pipe| { libc::close(pipe.in); }
            for out_pipe.each |pipe| { libc::close(pipe.out); }
            for err_pipe.each |pipe| { libc::close(pipe.out); }
        }

        Process {
            pid: res.pid,
            handle: res.handle,
            input: in_pipe.map(|pipe| pipe.out),
            output: out_pipe.map(|pipe| os::fdopen(pipe.in)),
            error: err_pipe.map(|pipe| os::fdopen(pipe.in)),
            exit_code: None,
        }
    }

    /// Returns the unique id of the process
    pub fn get_id(&self) -> pid_t { self.pid }

    fn input_fd(&mut self) -> c_int {
        match self.input {
            Some(fd) => fd,
            None => fail!("This Process's stdin was redirected to an \
                           existing file descriptor.")
        }
    }

    fn output_file(&mut self) -> *libc::FILE {
        match self.output {
            Some(file) => file,
            None => fail!("This Process's stdout was redirected to an \
                           existing file descriptor.")
        }
    }

    fn error_file(&mut self) -> *libc::FILE {
        match self.error {
            Some(file) => file,
            None => fail!("This Process's stderr was redirected to an \
                           existing file descriptor.")
        }
    }

    /**
     * Returns whether this process is reading its stdin from an existing file
     * descriptor rather than a pipe that was created specifically for this
     * process.
     *
     * If this method returns true then self.input() will fail.
     */
    pub fn input_redirected(&self) -> bool {
        self.input.is_none()
    }

    /**
     * Returns whether this process is writing its stdout to an existing file
     * descriptor rather than a pipe that was created specifically for this
     * process.
     *
     * If this method returns true then self.output() will fail.
     */
    pub fn output_redirected(&self) -> bool {
        self.output.is_none()
    }

    /**
     * Returns whether this process is writing its stderr to an existing file
     * descriptor rather than a pipe that was created specifically for this
     * process.
     *
     * If this method returns true then self.error() will fail.
     */
    pub fn error_redirected(&self) -> bool {
        self.error.is_none()
    }

    /**
     * Returns an io::Writer that can be used to write to this Process's stdin.
     *
     * Fails if this Process's stdin was redirected to an existing file descriptor.
     */
    pub fn input(&mut self) -> @io::Writer {
        // FIXME: the Writer can still be used after self is destroyed: #2625
       io::fd_writer(self.input_fd(), false)
    }

    /**
     * Returns an io::Reader that can be used to read from this Process's stdout.
     *
     * Fails if this Process's stdout was redirected to an existing file descriptor.
     */
    pub fn output(&mut self) -> @io::Reader {
        // FIXME: the Reader can still be used after self is destroyed: #2625
        io::FILE_reader(self.output_file(), false)
    }

    /**
     * Returns an io::Reader that can be used to read from this Process's stderr.
     *
     * Fails if this Process's stderr was redirected to an existing file descriptor.
     */
    pub fn error(&mut self) -> @io::Reader {
        // FIXME: the Reader can still be used after self is destroyed: #2625
        io::FILE_reader(self.error_file(), false)
    }

    /**
     * Closes the handle to the child process's stdin.
     *
     * If this process is reading its stdin from an existing file descriptor, then this
     * method does nothing.
     */
    pub fn close_input(&mut self) {
        match self.input {
            Some(-1) | None => (),
            Some(fd) => {
                unsafe {
                    libc::close(fd);
                }
                self.input = Some(-1);
            }
        }
    }

    fn close_outputs(&mut self) {
        fclose_and_null(&mut self.output);
        fclose_and_null(&mut self.error);

        fn fclose_and_null(f_opt: &mut Option<*libc::FILE>) {
            match *f_opt {
                Some(f) if !f.is_null() => {
                    unsafe {
                        libc::fclose(f);
                        *f_opt = Some(0 as *libc::FILE);
                    }
                },
                _ => ()
            }
        }
    }

    /**
     * Closes the handle to stdin, waits for the child process to terminate,
     * and returns the exit code.
     *
     * If the child has already been finished then the exit code is returned.
     */
    pub fn finish(&mut self) -> int {
        for self.exit_code.each |&code| {
            return code;
        }
        self.close_input();
        let code = waitpid(self.pid);
        self.exit_code = Some(code);
        return code;
    }

    /**
     * Closes the handle to stdin, waits for the child process to terminate, and reads
     * and returns all remaining output of stdout and stderr, along with the exit code.
     *
     * If the child has already been finished then the exit code and any remaining
     * unread output of stdout and stderr will be returned.
     *
     * This method will fail if the child process's stdout or stderr streams were
     * redirected to existing file descriptors.
     */
    pub fn finish_with_output(&mut self) -> ProcessOutput {
        let output_file = self.output_file();
        let error_file = self.error_file();

        // Spawn two entire schedulers to read both stdout and sterr
        // in parallel so we don't deadlock while blocking on one
        // or the other. FIXME (#2625): Surely there's a much more
        // clever way to do this.
        let (p, ch) = stream();
        let ch = SharedChan::new(ch);
        let ch_clone = ch.clone();
        do task::spawn_sched(task::SingleThreaded) {
            let errput = io::FILE_reader(error_file, false);
            ch.send((2, errput.read_whole_stream()));
        }
        do task::spawn_sched(task::SingleThreaded) {
            let output = io::FILE_reader(output_file, false);
            ch_clone.send((1, output.read_whole_stream()));
        }

        let status = self.finish();

        let (errs, outs) = match (p.recv(), p.recv()) {
            ((1, o), (2, e)) => (e, o),
            ((2, e), (1, o)) => (e, o),
            ((x, _), (y, _)) => {
                fail!("unexpected file numbers: %u, %u", x, y);
            }
        };

        return ProcessOutput {status: status,
                              output: outs,
                              error: errs};
    }

    fn destroy_internal(&mut self, force: bool) {
        // if the process has finished, and therefore had waitpid called,
        // and we kill it, then on unix we might ending up killing a
        // newer process that happens to have the same (re-used) id
        if self.exit_code.is_none() {
            killpid(self.pid, force);
            self.finish();
        }

        #[cfg(windows)]
        fn killpid(pid: pid_t, _force: bool) {
            unsafe {
                libc::funcs::extra::kernel32::TerminateProcess(
                    cast::transmute(pid), 1);
            }
        }

        #[cfg(unix)]
        fn killpid(pid: pid_t, force: bool) {
            let signal = if force {
                libc::consts::os::posix88::SIGKILL
            } else {
                libc::consts::os::posix88::SIGTERM
            };

            unsafe {
                libc::funcs::posix88::signal::kill(pid, signal as c_int);
            }
        }
    }

    /**
     * Terminates the process, giving it a chance to clean itself up if
     * this is supported by the operating system.
     *
     * On Posix OSs SIGTERM will be sent to the process. On Win32
     * TerminateProcess(..) will be called.
     */
    pub fn destroy(&mut self) { self.destroy_internal(false); }

    /**
     * Terminates the process as soon as possible without giving it a
     * chance to clean itself up.
     *
     * On Posix OSs SIGKILL will be sent to the process. On Win32
     * TerminateProcess(..) will be called.
     */
    pub fn force_destroy(&mut self) { self.destroy_internal(true); }
}

impl Drop for Process {
    fn finalize(&self) {
        // FIXME #4943: transmute is bad.
        let mut_self: &mut Process = unsafe { cast::transmute(self) };

        mut_self.finish();
        mut_self.close_outputs();
        free_handle(self.handle);
    }
}

struct SpawnProcessResult {
    pid: pid_t,
    handle: *(),
}

#[cfg(windows)]
fn spawn_process_os(prog: &str, args: &[~str],
                    env: Option<&[(~str, ~str)]>,
                    dir: Option<&Path>,
                    in_fd: c_int, out_fd: c_int, err_fd: c_int) -> SpawnProcessResult {

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

    use sys;
    use uint;

    unsafe {

        let mut si = zeroed_startupinfo();
        si.cb = sys::size_of::<STARTUPINFO>() as DWORD;
        si.dwFlags = STARTF_USESTDHANDLES;

        let cur_proc = GetCurrentProcess();

        let orig_std_in = get_osfhandle(in_fd) as HANDLE;
        if orig_std_in == INVALID_HANDLE_VALUE as HANDLE {
            fail!("failure in get_osfhandle: %s", os::last_os_error());
        }
        if DuplicateHandle(cur_proc, orig_std_in, cur_proc, &mut si.hStdInput,
                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
            fail!("failure in DuplicateHandle: %s", os::last_os_error());
        }

        let orig_std_out = get_osfhandle(out_fd) as HANDLE;
        if orig_std_out == INVALID_HANDLE_VALUE as HANDLE {
            fail!("failure in get_osfhandle: %s", os::last_os_error());
        }
        if DuplicateHandle(cur_proc, orig_std_out, cur_proc, &mut si.hStdOutput,
                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
            fail!("failure in DuplicateHandle: %s", os::last_os_error());
        }

        let orig_std_err = get_osfhandle(err_fd) as HANDLE;
        if orig_std_err == INVALID_HANDLE_VALUE as HANDLE {
            fail!("failure in get_osfhandle: %s", os::last_os_error());
        }
        if DuplicateHandle(cur_proc, orig_std_err, cur_proc, &mut si.hStdError,
                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
            fail!("failure in DuplicateHandle: %s", os::last_os_error());
        }

        let cmd = make_command_line(prog, args);
        let mut pi = zeroed_process_information();
        let mut create_err = None;

        do with_envp(env) |envp| {
            do with_dirp(dir) |dirp| {
                do str::as_c_str(cmd) |cmdp| {
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

        for create_err.each |msg| {
            fail!("failure in CreateProcess: %s", *msg);
        }

        // We close the thread handle because we don't care about keeping the thread id valid,
        // and we aren't keeping the thread handle around to be able to close it later. We don't
        // close the process handle however because we want the process id to stay valid at least
        // until the calling code closes the process handle.
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

    use uint;

    let mut cmd = ~"";
    append_arg(&mut cmd, prog);
    for args.each |arg| {
        cmd.push_char(' ');
        append_arg(&mut cmd, *arg);
    }
    return cmd;

    fn append_arg(cmd: &mut ~str, arg: &str) {
        let quote = arg.iter().any(|c| c == ' ' || c == '\t');
        if quote {
            cmd.push_char('"');
        }
        for uint::range(0, arg.len()) |i| {
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
                    env: Option<&[(~str, ~str)]>,
                    dir: Option<&Path>,
                    in_fd: c_int, out_fd: c_int, err_fd: c_int) -> SpawnProcessResult {

    use libc::funcs::posix88::unistd::{fork, dup2, close, chdir, execvp};
    use libc::funcs::bsd44::getdtablesize;

    mod rustrt {
        use libc::c_void;

        #[abi = "cdecl"]
        pub extern {
            unsafe fn rust_unset_sigprocmask();
            unsafe fn rust_set_environ(envp: *c_void);
        }
    }

    unsafe {

        let pid = fork();
        if pid < 0 {
            fail!("failure in fork: %s", os::last_os_error());
        } else if pid > 0 {
            return SpawnProcessResult {pid: pid, handle: ptr::null()};
        }

        rustrt::rust_unset_sigprocmask();

        if dup2(in_fd, 0) == -1 {
            fail!("failure in dup2(in_fd, 0): %s", os::last_os_error());
        }
        if dup2(out_fd, 1) == -1 {
            fail!("failure in dup2(out_fd, 1): %s", os::last_os_error());
        }
        if dup2(err_fd, 2) == -1 {
            fail!("failure in dup3(err_fd, 2): %s", os::last_os_error());
        }
        // close all other fds
        for int::range_rev(getdtablesize() as int - 1, 2) |fd| {
            close(fd as c_int);
        }

        do with_dirp(dir) |dirp| {
            if !dirp.is_null() && chdir(dirp) == -1 {
                fail!("failure in chdir: %s", os::last_os_error());
            }
        }

        do with_envp(env) |envp| {
            if !envp.is_null() {
                rustrt::rust_set_environ(envp);
            }
            do with_argv(prog, args) |argv| {
                execvp(*argv, argv);
                // execvp only returns if an error occurred
                fail!("failure in execvp: %s", os::last_os_error());
            }
        }
    }
}

#[cfg(unix)]
fn with_argv<T>(prog: &str, args: &[~str],
                cb: &fn(**libc::c_char) -> T) -> T {
    let mut argptrs = ~[str::as_c_str(prog, |b| b)];
    let mut tmps = ~[];
    for args.each |arg| {
        let t = @copy *arg;
        tmps.push(t);
        argptrs.push(str::as_c_str(*t, |b| b));
    }
    argptrs.push(ptr::null());
    vec::as_imm_buf(argptrs, |buf, _len| cb(buf))
}

#[cfg(unix)]
fn with_envp<T>(env: Option<&[(~str, ~str)]>, cb: &fn(*c_void) -> T) -> T {
    // On posixy systems we can pass a char** for envp, which is
    // a null-terminated array of "k=v\n" strings.
    match env {
      Some(es) => {
        let mut tmps = ~[];
        let mut ptrs = ~[];

        for es.each |&(k, v)| {
            let kv = @fmt!("%s=%s", k, v);
            tmps.push(kv);
            ptrs.push(str::as_c_str(*kv, |b| b));
        }

        ptrs.push(ptr::null());
        vec::as_imm_buf(ptrs, |p, _len|
            unsafe { cb(::cast::transmute(p)) }
        )
      }
      _ => cb(ptr::null())
    }
}

#[cfg(windows)]
fn with_envp<T>(env: Option<&[(~str, ~str)]>, cb: &fn(*mut c_void) -> T) -> T {
    // On win32 we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    match env {
      Some(es) => {
        let mut blk = ~[];
        for es.each |&(k, v)| {
            let kv = fmt!("%s=%s", k, v);
            blk.push_all(str::as_bytes_slice(kv));
            blk.push(0);
        }
        blk.push(0);
        vec::as_imm_buf(blk, |p, _len|
            unsafe { cb(::cast::transmute(p)) }
        )
      }
      _ => cb(ptr::mut_null())
    }
}

fn with_dirp<T>(d: Option<&Path>,
                cb: &fn(*libc::c_char) -> T) -> T {
    match d {
      Some(dir) => str::as_c_str(dir.to_str(), cb),
      None => cb(ptr::null())
    }
}

#[cfg(windows)]
priv fn free_handle(handle: *()) {
    unsafe {
        libc::funcs::extra::kernel32::CloseHandle(cast::transmute(handle));
    }
}

#[cfg(unix)]
priv fn free_handle(_handle: *()) {
    // unix has no process handle object, just a pid
}

/**
 * Spawns a process and waits for it to terminate. The process will
 * inherit the current stdin/stdout/stderr file descriptors.
 *
 * # Arguments
 *
 * * prog - The path to an executable
 * * args - Vector of arguments to pass to the child process
 *
 * # Return value
 *
 * The process's exit code
 */
pub fn process_status(prog: &str, args: &[~str]) -> int {
    let mut prog = Process::new(prog, args, ProcessOptions {
        env: None,
        dir: None,
        in_fd: Some(0),
        out_fd: Some(1),
        err_fd: Some(2)
    });
    prog.finish()
}

/**
 * Spawns a process, records all its output, and waits for it to terminate.
 *
 * # Arguments
 *
 * * prog - The path to an executable
 * * args - Vector of arguments to pass to the child process
 *
 * # Return value
 *
 * The process's stdout/stderr output and exit code.
 */
pub fn process_output(prog: &str, args: &[~str]) -> ProcessOutput {
    let mut prog = Process::new(prog, args, ProcessOptions::new());
    prog.finish_with_output()
}

/**
 * Waits for a process to exit and returns the exit code, failing
 * if there is no process with the specified id.
 *
 * Note that this is private to avoid race conditions on unix where if
 * a user calls waitpid(some_process.get_id()) then some_process.finish()
 * and some_process.destroy() and some_process.finalize() will then either
 * operate on a none-existant process or, even worse, on a newer process
 * with the same id.
 */
priv fn waitpid(pid: pid_t) -> int {
    return waitpid_os(pid);

    #[cfg(windows)]
    fn waitpid_os(pid: pid_t) -> int {

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

            let proc = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_INFORMATION, FALSE, pid as DWORD);
            if proc.is_null() {
                fail!("failure in OpenProcess: %s", os::last_os_error());
            }

            loop {
                let mut status = 0;
                if GetExitCodeProcess(proc, &mut status) == FALSE {
                    CloseHandle(proc);
                    fail!("failure in GetExitCodeProcess: %s", os::last_os_error());
                }
                if status != STILL_ACTIVE {
                    CloseHandle(proc);
                    return status as int;
                }
                if WaitForSingleObject(proc, INFINITE) == WAIT_FAILED {
                    CloseHandle(proc);
                    fail!("failure in WaitForSingleObject: %s", os::last_os_error());
                }
            }
        }
    }

    #[cfg(unix)]
    fn waitpid_os(pid: pid_t) -> int {

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
            fail!("failure in waitpid: %s", os::last_os_error());
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
    use io;
    use libc::{c_int};
    use option::{Option, None, Some};
    use os;
    use path::Path;
    use run;
    use str;

    #[test]
    #[cfg(windows)]
    fn test_make_command_line() {
        assert_eq!(
            run::make_command_line("prog", [~"aaa", ~"bbb", ~"ccc"]),
            ~"prog aaa bbb ccc"
        );
        assert_eq!(
            run::make_command_line("C:\\Program Files\\blah\\blah.exe", [~"aaa"]),
            ~"\"C:\\Program Files\\blah\\blah.exe\" aaa"
        );
        assert_eq!(
            run::make_command_line("C:\\Program Files\\test", [~"aa\"bb"]),
            ~"\"C:\\Program Files\\test\" aa\\\"bb"
        );
        assert_eq!(
            run::make_command_line("echo", [~"a b c"]),
            ~"echo \"a b c\""
        );
    }

    #[test]
    fn test_process_status() {
        assert_eq!(run::process_status("false", []), 1);
        assert_eq!(run::process_status("true", []), 0);
    }

    #[test]
    fn test_process_output_output() {

        let run::ProcessOutput {status, output, error}
             = run::process_output("echo", [~"hello"]);
        let output_str = str::from_bytes(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        assert_eq!(error, ~[]);
    }

    #[test]
    fn test_process_output_error() {

        let run::ProcessOutput {status, output, error}
             = run::process_output("mkdir", [~"."]);

        assert_eq!(status, 1);
        assert_eq!(output, ~[]);
        assert!(!error.is_empty());
    }

    #[test]
    fn test_pipes() {

        let pipe_in = os::pipe();
        let pipe_out = os::pipe();
        let pipe_err = os::pipe();

        let mut proc = run::Process::new("cat", [], run::ProcessOptions {
            dir: None,
            env: None,
            in_fd: Some(pipe_in.in),
            out_fd: Some(pipe_out.out),
            err_fd: Some(pipe_err.out)
        });

        assert!(proc.input_redirected());
        assert!(proc.output_redirected());
        assert!(proc.error_redirected());

        os::close(pipe_in.in);
        os::close(pipe_out.out);
        os::close(pipe_err.out);

        let expected = ~"test";
        writeclose(pipe_in.out, expected);
        let actual = readclose(pipe_out.in);
        readclose(pipe_err.in);
        proc.finish();

        assert_eq!(expected, actual);
    }

    fn writeclose(fd: c_int, s: &str) {
        let writer = io::fd_writer(fd, false);
        writer.write_str(s);
        os::close(fd);
    }

    fn readclose(fd: c_int) -> ~str {
        unsafe {
            let file = os::fdopen(fd);
            let reader = io::FILE_reader(file, false);
            let buf = reader.read_whole_stream();
            os::fclose(file);
            str::from_bytes(buf)
        }
    }

    #[test]
    fn test_finish_once() {
        let mut prog = run::Process::new("false", [], run::ProcessOptions::new());
        assert_eq!(prog.finish(), 1);
    }

    #[test]
    fn test_finish_twice() {
        let mut prog = run::Process::new("false", [], run::ProcessOptions::new());
        assert_eq!(prog.finish(), 1);
        assert_eq!(prog.finish(), 1);
    }

    #[test]
    fn test_finish_with_output_once() {

        let mut prog = run::Process::new("echo", [~"hello"], run::ProcessOptions::new());
        let run::ProcessOutput {status, output, error}
            = prog.finish_with_output();
        let output_str = str::from_bytes(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        assert_eq!(error, ~[]);
    }

    #[test]
    fn test_finish_with_output_twice() {

        let mut prog = run::Process::new("echo", [~"hello"], run::ProcessOptions::new());
        let run::ProcessOutput {status, output, error}
            = prog.finish_with_output();

        let output_str = str::from_bytes(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        assert_eq!(error, ~[]);

        let run::ProcessOutput {status, output, error}
            = prog.finish_with_output();

        assert_eq!(status, 0);
        assert_eq!(output, ~[]);
        assert_eq!(error, ~[]);
    }

    #[test]
    #[should_fail]
    #[cfg(not(windows))]
    fn test_finish_with_output_redirected() {
        let mut prog = run::Process::new("echo", [~"hello"], run::ProcessOptions {
            env: None,
            dir: None,
            in_fd: Some(0),
            out_fd: Some(1),
            err_fd: Some(2)
        });
        // this should fail because it is not valid to read the output when it was redirected
        prog.finish_with_output();
    }

    #[cfg(unix)]
    fn run_pwd(dir: Option<&Path>) -> run::Process {
        run::Process::new("pwd", [], run::ProcessOptions {
            dir: dir,
            .. run::ProcessOptions::new()
        })
    }

    #[cfg(windows)]
    fn run_pwd(dir: Option<&Path>) -> run::Process {
        run::Process::new("cmd", [~"/c", ~"cd"], run::ProcessOptions {
            dir: dir,
            .. run::ProcessOptions::new()
        })
    }

    #[test]
    fn test_keep_current_working_dir() {
        let mut prog = run_pwd(None);

        let output = str::from_bytes(prog.finish_with_output().output);
        let parent_dir = os::getcwd().normalize();
        let child_dir = Path(output.trim()).normalize();

        let parent_stat = parent_dir.stat().unwrap();
        let child_stat = child_dir.stat().unwrap();

        assert_eq!(parent_stat.st_dev, child_stat.st_dev);
        assert_eq!(parent_stat.st_ino, child_stat.st_ino);
    }

    #[test]
    fn test_change_working_directory() {
        // test changing to the parent of os::getcwd() because we know
        // the path exists (and os::getcwd() is not expected to be root)
        let parent_dir = os::getcwd().dir_path().normalize();
        let mut prog = run_pwd(Some(&parent_dir));

        let output = str::from_bytes(prog.finish_with_output().output);
        let child_dir = Path(output.trim()).normalize();

        let parent_stat = parent_dir.stat().unwrap();
        let child_stat = child_dir.stat().unwrap();

        assert_eq!(parent_stat.st_dev, child_stat.st_dev);
        assert_eq!(parent_stat.st_ino, child_stat.st_ino);
    }

    #[cfg(unix)]
    fn run_env(env: Option<&[(~str, ~str)]>) -> run::Process {
        run::Process::new("env", [], run::ProcessOptions {
            env: env,
            .. run::ProcessOptions::new()
        })
    }

    #[cfg(windows)]
    fn run_env(env: Option<&[(~str, ~str)]>) -> run::Process {
        run::Process::new("cmd", [~"/c", ~"set"], run::ProcessOptions {
            env: env,
            .. run::ProcessOptions::new()
        })
    }

    #[test]
    fn test_inherit_env() {

        let mut prog = run_env(None);
        let output = str::from_bytes(prog.finish_with_output().output);

        for os::env().each |&(k, v)| {
            // don't check windows magical empty-named variables
            assert!(k.is_empty() || output.contains(fmt!("%s=%s", k, v)));
        }
    }

    #[test]
    fn test_add_to_env() {

        let mut new_env = os::env();
        new_env.push((~"RUN_TEST_NEW_ENV", ~"123"));

        let mut prog = run_env(Some(new_env.slice(0, new_env.len())));
        let output = str::from_bytes(prog.finish_with_output().output);

        assert!(output.contains("RUN_TEST_NEW_ENV=123"));
    }
}
