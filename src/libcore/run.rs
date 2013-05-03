// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Process spawning
use cast;
use io;
use libc;
use libc::{pid_t, c_void, c_int};
use comm::{stream, SharedChan, GenericChan, GenericPort};
use option::{Some, None};
use os;
use prelude::*;
use ptr;
use str;
use task;
use vec;

/// A value representing a child process
pub struct Program {
    priv pid: pid_t,
    priv handle: *(),
    priv in_fd: c_int,
    priv out_file: *libc::FILE,
    priv err_file: *libc::FILE,
    priv finished: bool,
}

impl Drop for Program {
    fn finalize(&self) {
        // FIXME #4943: transmute is bad.
        let mut_self: &mut Program = unsafe { cast::transmute(self) };

        mut_self.finish();
        mut_self.close_outputs();
        free_handle(self.handle);
    }
}

pub impl Program {

    /// Returns the process id of the program
    fn get_id(&mut self) -> pid_t { self.pid }

    /// Returns an io::Writer that can be used to write to stdin
    fn input(&mut self) -> @io::Writer {
        io::fd_writer(self.in_fd, false)
    }

    /// Returns an io::Reader that can be used to read from stdout
    fn output(&mut self) -> @io::Reader {
        io::FILE_reader(self.out_file, false)
    }

    /// Returns an io::Reader that can be used to read from stderr
    fn err(&mut self) -> @io::Reader {
        io::FILE_reader(self.err_file, false)
    }

    /// Closes the handle to the child processes standard input
    fn close_input(&mut self) {
        let invalid_fd = -1i32;
        if self.in_fd != invalid_fd {
            unsafe {
                libc::close(self.in_fd);
            }
            self.in_fd = invalid_fd;
        }
    }

    priv fn close_outputs(&mut self) {
        unsafe {
            fclose_and_null(&mut self.out_file);
            fclose_and_null(&mut self.err_file);
        }
    }

    /**
     * Waits for the child process to terminate. Closes the handle
     * to stdin if necessary.
     */
    fn finish(&mut self) -> int {
        if self.finished { return 0; }
        self.finished = true;
        self.close_input();
        return waitpid(self.pid);
    }

    priv fn destroy_internal(&mut self, force: bool) {
        killpid(self.pid, force);
        self.finish();
        self.close_outputs();

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
     * Terminate the program, giving it a chance to clean itself up if
     * this is supported by the operating system.
     *
     * On Posix OSs SIGTERM will be sent to the process. On Win32
     * TerminateProcess(..) will be called.
     */
    fn destroy(&mut self) { self.destroy_internal(false); }

    /**
     * Terminate the program as soon as possible without giving it a
     * chance to clean itself up.
     *
     * On Posix OSs SIGKILL will be sent to the process. On Win32
     * TerminateProcess(..) will be called.
     */
    fn force_destroy(&mut self) { self.destroy_internal(true); }
}


/**
 * Run a program, providing stdin, stdout and stderr handles
 *
 * # Arguments
 *
 * * prog - The path to an executable
 * * args - Vector of arguments to pass to the child process
 * * env - optional env-modification for child
 * * dir - optional dir to run child in (default current dir)
 * * in_fd - A file descriptor for the child to use as std input
 * * out_fd - A file descriptor for the child to use as std output
 * * err_fd - A file descriptor for the child to use as std error
 *
 * # Return value
 *
 * The process id of the spawned process
 */
pub fn spawn_process(prog: &str, args: &[~str],
                     env: &Option<~[(~str,~str)]>,
                     dir: &Option<~str>,
                     in_fd: c_int, out_fd: c_int, err_fd: c_int) -> pid_t {

    let res = spawn_process_internal(prog, args, env, dir, in_fd, out_fd, err_fd);
    free_handle(res.handle);
    return res.pid;
}

struct RunProgramResult {
    // the process id of the program (this should never be negative)
    pid: pid_t,
    // a handle to the process - on unix this will always be NULL, but on windows it will be a
    // HANDLE to the process, which will prevent the pid being re-used until the handle is closed.
    handle: *(),
}

#[cfg(windows)]
fn spawn_process_internal(prog: &str, args: &[~str],
                          env: &Option<~[(~str,~str)]>,
                          dir: &Option<~str>,
                          in_fd: c_int, out_fd: c_int, err_fd: c_int) -> RunProgramResult {

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

    unsafe {

        let mut si = zeroed_startupinfo();
        si.cb = sys::size_of::<STARTUPINFO>() as DWORD;
        si.dwFlags = STARTF_USESTDHANDLES;

        let cur_proc = GetCurrentProcess();

        let orig_std_in = get_osfhandle(if in_fd > 0 { in_fd } else { 0 }) as HANDLE;
        if orig_std_in == INVALID_HANDLE_VALUE as HANDLE {
            fail!(fmt!("failure in get_osfhandle: %s", os::last_os_error()));
        }
        if DuplicateHandle(cur_proc, orig_std_in, cur_proc, &mut si.hStdInput,
                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
            fail!(fmt!("failure in DuplicateHandle: %s", os::last_os_error()));
        }

        let orig_std_out = get_osfhandle(if out_fd > 0 { out_fd } else { 1 }) as HANDLE;
        if orig_std_out == INVALID_HANDLE_VALUE as HANDLE {
            fail!(fmt!("failure in get_osfhandle: %s", os::last_os_error()));
        }
        if DuplicateHandle(cur_proc, orig_std_out, cur_proc, &mut si.hStdOutput,
                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
            fail!(fmt!("failure in DuplicateHandle: %s", os::last_os_error()));
        }

        let orig_std_err = get_osfhandle(if err_fd > 0 { err_fd } else { 2 }) as HANDLE;
        if orig_std_err as HANDLE == INVALID_HANDLE_VALUE as HANDLE {
            fail!(fmt!("failure in get_osfhandle: %s", os::last_os_error()));
        }
        if DuplicateHandle(cur_proc, orig_std_err, cur_proc, &mut si.hStdError,
                           0, TRUE, DUPLICATE_SAME_ACCESS) == FALSE {
            fail!(fmt!("failure in DuplicateHandle: %s", os::last_os_error()));
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
            fail!(fmt!("failure in CreateProcess: %s", *msg));
        }

        // We close the thread handle because we don't care about keeping the thread id valid,
        // and we aren't keeping the thread handle around to be able to close it later. We don't
        // close the process handle however because we want the process id to stay valid at least
        // until the calling code closes the process handle.
        CloseHandle(pi.hThread);

        RunProgramResult {
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
    for args.each |arg| {
        cmd.push_char(' ');
        append_arg(&mut cmd, *arg);
    }
    return cmd;

    fn append_arg(cmd: &mut ~str, arg: &str) {
        let quote = arg.any(|c| c == ' ' || c == '\t');
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
fn spawn_process_internal(prog: &str, args: &[~str],
                          env: &Option<~[(~str,~str)]>,
                          dir: &Option<~str>,
                          in_fd: c_int, out_fd: c_int, err_fd: c_int) -> RunProgramResult {

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
            fail!(fmt!("failure in fork: %s", os::last_os_error()));
        } else if pid > 0 {
            return RunProgramResult {pid: pid, handle: ptr::null()};
        }

        rustrt::rust_unset_sigprocmask();

        if in_fd > 0 && dup2(in_fd, 0) == -1 {
            fail!(fmt!("failure in dup2(in_fd, 0): %s", os::last_os_error()));
        }
        if out_fd > 0 && dup2(out_fd, 1) == -1 {
            fail!(fmt!("failure in dup2(out_fd, 1): %s", os::last_os_error()));
        }
        if err_fd > 0 && dup2(err_fd, 2) == -1 {
            fail!(fmt!("failure in dup3(err_fd, 2): %s", os::last_os_error()));
        }
        // close all other fds
        for int::range_rev(getdtablesize() as int - 1, 2) |fd| {
            close(fd as c_int);
        }

        for dir.each |dir| {
            do str::as_c_str(*dir) |dirp| {
                if chdir(dirp) == -1 {
                    fail!(fmt!("failure in chdir: %s", os::last_os_error()));
                }
            }
        }

        do with_envp(env) |envp| {
            if !envp.is_null() {
                rustrt::rust_set_environ(envp);
            }
            do with_argv(prog, args) |argv| {
                execvp(*argv, argv);
                // execvp only returns if an error occurred
                fail!(fmt!("failure in execvp: %s", os::last_os_error()));
            }
        }
    }
}

#[cfg(unix)]
fn with_argv<T>(prog: &str, args: &[~str],
                cb: &fn(**libc::c_char) -> T) -> T {
    let mut argptrs = str::as_c_str(prog, |b| ~[b]);
    let mut tmps = ~[];
    for vec::each(args) |arg| {
        let t = @copy *arg;
        tmps.push(t);
        argptrs.push_all(str::as_c_str(*t, |b| ~[b]));
    }
    argptrs.push(ptr::null());
    vec::as_imm_buf(argptrs, |buf, _len| cb(buf))
}

#[cfg(unix)]
fn with_envp<T>(env: &Option<~[(~str,~str)]>,
                cb: &fn(*c_void) -> T) -> T {
    // On posixy systems we can pass a char** for envp, which is
    // a null-terminated array of "k=v\n" strings.
    match *env {
      Some(ref es) if !vec::is_empty(*es) => {
        let mut tmps = ~[];
        let mut ptrs = ~[];

        for vec::each(*es) |e| {
            let (k,v) = copy *e;
            let t = @(fmt!("%s=%s", k, v));
            tmps.push(t);
            ptrs.push_all(str::as_c_str(*t, |b| ~[b]));
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
fn with_envp<T>(env: &Option<~[(~str,~str)]>,
                cb: &fn(*mut c_void) -> T) -> T {
    // On win32 we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    unsafe {
        match *env {
          Some(ref es) if !vec::is_empty(*es) => {
            let mut blk : ~[u8] = ~[];
            for vec::each(*es) |e| {
                let (k,v) = copy *e;
                let t = fmt!("%s=%s", k, v);
                let mut v : ~[u8] = ::cast::transmute(t);
                blk += v;
                ::cast::forget(v);
            }
            blk += ~[0_u8];
            vec::as_imm_buf(blk, |p, _len| cb(::cast::transmute(p)))
          }
          _ => cb(ptr::mut_null())
        }
    }
}

#[cfg(windows)]
fn with_dirp<T>(d: &Option<~str>,
                cb: &fn(*libc::c_char) -> T) -> T {
    match *d {
      Some(ref dir) => str::as_c_str(*dir, cb),
      None => cb(ptr::null())
    }
}

/// helper function that closes non-NULL files and then makes them NULL
priv unsafe fn fclose_and_null(f: &mut *libc::FILE) {
    if *f != 0 as *libc::FILE {
        libc::fclose(*f);
        *f = 0 as *libc::FILE;
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
 * Spawns a process and waits for it to terminate
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
pub fn run_program(prog: &str, args: &[~str]) -> int {
    let res = spawn_process_internal(prog, args, &None, &None,
                                     0i32, 0i32, 0i32);
    let code = waitpid(res.pid);
    free_handle(res.handle);
    return code;
}

/**
 * Spawns a process and returns a Program
 *
 * The returned value is a <Program> object that can be used for sending and
 * receiving data over the standard file descriptors.  The class will ensure
 * that file descriptors are closed properly.
 *
 * # Arguments
 *
 * * prog - The path to an executable
 * * args - Vector of arguments to pass to the child process
 *
 * # Return value
 *
 * A <Program> object
 */
pub fn start_program(prog: &str, args: &[~str]) -> Program {
    let pipe_input = os::pipe();
    let pipe_output = os::pipe();
    let pipe_err = os::pipe();
    let res =
        spawn_process_internal(prog, args, &None, &None,
                               pipe_input.in, pipe_output.out,
                               pipe_err.out);

    unsafe {
        libc::close(pipe_input.in);
        libc::close(pipe_output.out);
        libc::close(pipe_err.out);
    }

    Program {
        pid: res.pid,
        handle: res.handle,
        in_fd: pipe_input.out,
        out_file: os::fdopen(pipe_output.in),
        err_file: os::fdopen(pipe_err.in),
        finished: false,
    }
}

fn read_all(rd: @io::Reader) -> ~str {
    let buf = io::with_bytes_writer(|wr| {
        let mut bytes = [0, ..4096];
        while !rd.eof() {
            let nread = rd.read(bytes, bytes.len());
            wr.write(bytes.slice(0, nread));
        }
    });
    str::from_bytes(buf)
}

pub struct ProgramOutput {status: int, out: ~str, err: ~str}

/**
 * Spawns a process, waits for it to exit, and returns the exit code, and
 * contents of stdout and stderr.
 *
 * # Arguments
 *
 * * prog - The path to an executable
 * * args - Vector of arguments to pass to the child process
 *
 * # Return value
 *
 * A record, {status: int, out: str, err: str} containing the exit code,
 * the contents of stdout and the contents of stderr.
 */
pub fn program_output(prog: &str, args: &[~str]) -> ProgramOutput {
    let pipe_in = os::pipe();
    let pipe_out = os::pipe();
    let pipe_err = os::pipe();
    let res = spawn_process_internal(prog, args, &None, &None,
                                     pipe_in.in, pipe_out.out, pipe_err.out);

    os::close(pipe_in.in);
    os::close(pipe_out.out);
    os::close(pipe_err.out);
    os::close(pipe_in.out);

    // Spawn two entire schedulers to read both stdout and sterr
    // in parallel so we don't deadlock while blocking on one
    // or the other. FIXME (#2625): Surely there's a much more
    // clever way to do this.
    let (p, ch) = stream();
    let ch = SharedChan::new(ch);
    let ch_clone = ch.clone();
    do task::spawn_sched(task::SingleThreaded) {
        let errput = readclose(pipe_err.in);
        ch.send((2, errput));
    };
    do task::spawn_sched(task::SingleThreaded) {
        let output = readclose(pipe_out.in);
        ch_clone.send((1, output));
    };

    let status = waitpid(res.pid);
    free_handle(res.handle);

    let mut errs = ~"";
    let mut outs = ~"";
    let mut count = 2;
    while count > 0 {
        let stream = p.recv();
        match stream {
            (1, copy s) => {
                outs = s;
            }
            (2, copy s) => {
                errs = s;
            }
            (n, _) => {
                fail!(fmt!("program_output received an unexpected file \
                           number: %u", n));
            }
        };
        count -= 1;
    };
    return ProgramOutput {status: status,
                          out: outs,
                          err: errs};
}

pub fn writeclose(fd: c_int, s: ~str) {
    use io::WriterUtil;

    error!("writeclose %d, %s", fd as int, s);
    let writer = io::fd_writer(fd, false);
    writer.write_str(s);

    os::close(fd);
}

pub fn readclose(fd: c_int) -> ~str {
    unsafe {
        let file = os::fdopen(fd);
        let reader = io::FILE_reader(file, false);
        let buf = io::with_bytes_writer(|writer| {
            let mut bytes = [0, ..4096];
            while !reader.eof() {
                let nread = reader.read(bytes, bytes.len());
                writer.write(bytes.slice(0, nread));
            }
        });
        os::fclose(file);
        str::from_bytes(buf)
    }
}

/**
 * Waits for a process to exit and returns the exit code, failing
 * if there is no process with the specified id.
 */
pub fn waitpid(pid: pid_t) -> int {
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
                fail!(fmt!("failure in OpenProcess: %s", os::last_os_error()));
            }

            loop {
                let mut status = 0;
                if GetExitCodeProcess(proc, &mut status) == FALSE {
                    CloseHandle(proc);
                    fail!(fmt!("failure in GetExitCodeProcess: %s", os::last_os_error()));
                }
                if status != STILL_ACTIVE {
                    CloseHandle(proc);
                    return status as int;
                }
                if WaitForSingleObject(proc, INFINITE) == WAIT_FAILED {
                    CloseHandle(proc);
                    fail!(fmt!("failure in WaitForSingleObject: %s", os::last_os_error()));
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
            fail!(fmt!("failure in waitpid: %s", os::last_os_error()));
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
    use libc;
    use option::None;
    use os;
    use run::{readclose, writeclose};
    use run;

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

    // Regression test for memory leaks
    #[test]
    fn test_leaks() {
        run::run_program("echo", []);
        run::start_program("echo", []);
        run::program_output("echo", []);
    }

    #[test]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_pipes() {
        let pipe_in = os::pipe();
        let pipe_out = os::pipe();
        let pipe_err = os::pipe();

        let pid =
            run::spawn_process(
                "cat", [], &None, &None,
                pipe_in.in, pipe_out.out, pipe_err.out);
        os::close(pipe_in.in);
        os::close(pipe_out.out);
        os::close(pipe_err.out);

        if pid == -1i32 { fail!(); }
        let expected = ~"test";
        writeclose(pipe_in.out, copy expected);
        let actual = readclose(pipe_out.in);
        readclose(pipe_err.in);
        run::waitpid(pid);

        debug!(copy expected);
        debug!(copy actual);
        assert!((expected == actual));
    }

    #[test]
    fn waitpid() {
        let pid = run::spawn_process("false", [],
                                     &None, &None,
                                     0i32, 0i32, 0i32);
        let status = run::waitpid(pid);
        assert!(status == 1);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn waitpid_non_existant_pid() {
        run::waitpid(123456789); // assume that this pid doesn't exist
    }

    #[test]
    fn test_destroy_once() {
        let mut p = run::start_program("echo", []);
        p.destroy(); // this shouldn't crash (and nor should the destructor)
    }

    #[test]
    fn test_destroy_twice() {
        let mut p = run::start_program("echo", []);
        p.destroy(); // this shouldnt crash...
        p.destroy(); // ...and nor should this (and nor should the destructor)
    }

    fn test_destroy_actually_kills(force: bool) {

        #[cfg(unix)]
        static BLOCK_COMMAND: &'static str = "cat";

        #[cfg(windows)]
        static BLOCK_COMMAND: &'static str = "cmd";

        #[cfg(unix)]
        fn process_exists(pid: libc::pid_t) -> bool {
            run::program_output("ps", [~"-p", pid.to_str()]).out.contains(pid.to_str())
        }

        #[cfg(windows)]
        fn process_exists(pid: libc::pid_t) -> bool {

            use libc::types::os::arch::extra::DWORD;
            use libc::funcs::extra::kernel32::{CloseHandle, GetExitCodeProcess, OpenProcess};
            use libc::consts::os::extra::{FALSE, PROCESS_QUERY_INFORMATION, STILL_ACTIVE };

            unsafe {
                let proc = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, pid as DWORD);
                if proc.is_null() {
                    return false;
                }
                // proc will be non-null if the process is alive, or if it died recently
                let mut status = 0;
                GetExitCodeProcess(proc, &mut status);
                CloseHandle(proc);
                return status == STILL_ACTIVE;
            }
        }

        // this program will stay alive indefinitely trying to read from stdin
        let mut p = run::start_program(BLOCK_COMMAND, []);

        assert!(process_exists(p.get_id()));

        if force {
            p.force_destroy();
        } else {
            p.destroy();
        }

        assert!(!process_exists(p.get_id()));
    }

    #[test]
    fn test_unforced_destroy_actually_kills() {
        test_destroy_actually_kills(false);
    }

    #[test]
    fn test_forced_destroy_actually_kills() {
        test_destroy_actually_kills(true);
    }
}
