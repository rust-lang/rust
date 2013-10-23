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

use cell::Cell;
use comm::{stream, SharedChan};
use libc::{pid_t, c_int};
use libc;
use prelude::*;
use rt::io::native::process;
use rt::io;
use task;

/**
 * A value representing a child process.
 *
 * The lifetime of this value is linked to the lifetime of the actual
 * process - the Process destructor calls self.finish() which waits
 * for the process to terminate.
 */
pub struct Process {
    priv inner: process::Process,
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
    env: Option<~[(~str, ~str)]>,

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
     * If this is None then a new pipe will be created for the new program's
     * output and Process.output() will provide a Reader to read from this pipe.
     *
     * If this is Some(file-descriptor) then the new process will write its output
     * to the given file descriptor, Process.output_redirected() will return
     * true, and Process.output() will fail.
     */
    out_fd: Option<c_int>,

    /**
     * If this is None then a new pipe will be created for the new program's
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
    pub fn new(prog: &str, args: &[~str], options: ProcessOptions) -> Process {
        let ProcessOptions { env, dir, in_fd, out_fd, err_fd } = options;
        let inner = process::Process::new(prog, args, env, dir,
                                          in_fd, out_fd, err_fd);
        Process { inner: inner }
    }

    /// Returns the unique id of the process
    pub fn get_id(&self) -> pid_t { self.inner.id() }

    /**
     * Returns an io::Writer that can be used to write to this Process's stdin.
     *
     * Fails if there is no stdin available (it's already been removed by
     * take_input)
     */
    pub fn input<'a>(&'a mut self) -> &'a mut io::Writer { self.inner.input() }

    /**
     * Returns an io::Reader that can be used to read from this Process's stdout.
     *
     * Fails if there is no stdout available (it's already been removed by
     * take_output)
     */
    pub fn output<'a>(&'a mut self) -> &'a mut io::Reader { self.inner.output() }

    /**
     * Returns an io::Reader that can be used to read from this Process's stderr.
     *
     * Fails if there is no stderr available (it's already been removed by
     * take_error)
     */
    pub fn error<'a>(&'a mut self) -> &'a mut io::Reader { self.inner.error() }

    /**
     * Closes the handle to the child process's stdin.
     */
    pub fn close_input(&mut self) {
        self.inner.take_input();
    }

    fn close_outputs(&mut self) {
        self.inner.take_output();
        self.inner.take_error();
    }

    /**
     * Closes the handle to stdin, waits for the child process to terminate,
     * and returns the exit code.
     *
     * If the child has already been finished then the exit code is returned.
     */
    pub fn finish(&mut self) -> int { self.inner.wait() }

    /**
     * Closes the handle to stdin, waits for the child process to terminate, and
     * reads and returns all remaining output of stdout and stderr, along with
     * the exit code.
     *
     * If the child has already been finished then the exit code and any
     * remaining unread output of stdout and stderr will be returned.
     *
     * This method will fail if the child process's stdout or stderr streams
     * were redirected to existing file descriptors.
     */
    pub fn finish_with_output(&mut self) -> ProcessOutput {
        self.inner.take_input(); // close stdin
        let output = Cell::new(self.inner.take_output());
        let error = Cell::new(self.inner.take_error());

        fn read_everything(r: &mut io::Reader) -> ~[u8] {
            let mut ret = ~[];
            let mut buf = [0, ..1024];
            loop {
                match r.read(buf) {
                    Some(n) => { ret.push_all(buf.slice_to(n)); }
                    None => break
                }
            }
            return ret;
        }

        // Spawn two entire schedulers to read both stdout and sterr
        // in parallel so we don't deadlock while blocking on one
        // or the other. FIXME (#2625): Surely there's a much more
        // clever way to do this.
        let (p, ch) = stream();
        let ch = SharedChan::new(ch);
        let ch_clone = ch.clone();
        do task::spawn_sched(task::SingleThreaded) {
            match error.take() {
                Some(ref mut e) => ch.send((2, read_everything(*e))),
                None => ch.send((2, ~[]))
            }
        }
        do task::spawn_sched(task::SingleThreaded) {
            match output.take() {
                Some(ref mut e) => ch_clone.send((1, read_everything(*e))),
                None => ch_clone.send((1, ~[]))
            }
        }

        let status = self.finish();

        let (errs, outs) = match (p.recv(), p.recv()) {
            ((1, o), (2, e)) => (e, o),
            ((2, e), (1, o)) => (e, o),
            ((x, _), (y, _)) => {
                fail!("unexpected file numbers: {}, {}", x, y);
            }
        };

        return ProcessOutput {status: status,
                              output: outs,
                              error: errs};
    }

    /**
     * Terminates the process, giving it a chance to clean itself up if
     * this is supported by the operating system.
     *
     * On Posix OSs SIGTERM will be sent to the process. On Win32
     * TerminateProcess(..) will be called.
     */
    pub fn destroy(&mut self) {
        self.inner.signal(io::process::PleaseExitSignal);
        self.finish();
    }

    /**
     * Terminates the process as soon as possible without giving it a
     * chance to clean itself up.
     *
     * On Posix OSs SIGKILL will be sent to the process. On Win32
     * TerminateProcess(..) will be called.
     */
    pub fn force_destroy(&mut self) {
        self.inner.signal(io::process::MustDieSignal);
        self.finish();
    }
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
#[fixed_stack_segment] #[inline(never)]
pub fn process_status(prog: &str, args: &[~str]) -> int {
    let mut prog = Process::new(prog, args, ProcessOptions {
        env: None,
        dir: None,
        in_fd: Some(unsafe { libc::dup(libc::STDIN_FILENO) }),
        out_fd: Some(unsafe { libc::dup(libc::STDOUT_FILENO) }),
        err_fd: Some(unsafe { libc::dup(libc::STDERR_FILENO) })
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

#[cfg(test)]
mod tests {
    use libc::c_int;
    use option::{Option, None, Some};
    use os;
    use path::Path;
    use run;
    use str;
    use unstable::running_on_valgrind;
    use rt::io::native::file;
    use rt::io::{Writer, Reader};

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_process_status() {
        assert_eq!(run::process_status("false", []), 1);
        assert_eq!(run::process_status("true", []), 0);
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_process_status() {
        assert_eq!(run::process_status("/system/bin/sh", [~"-c",~"false"]), 1);
        assert_eq!(run::process_status("/system/bin/sh", [~"-c",~"true"]), 0);
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_process_output_output() {

        let run::ProcessOutput {status, output, error}
             = run::process_output("echo", [~"hello"]);
        let output_str = str::from_utf8(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, ~[]);
        }
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_process_output_output() {

        let run::ProcessOutput {status, output, error}
             = run::process_output("/system/bin/sh", [~"-c",~"echo hello"]);
        let output_str = str::from_utf8(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, ~[]);
        }
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_process_output_error() {

        let run::ProcessOutput {status, output, error}
             = run::process_output("mkdir", [~"."]);

        assert_eq!(status, 1);
        assert_eq!(output, ~[]);
        assert!(!error.is_empty());
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_process_output_error() {

        let run::ProcessOutput {status, output, error}
             = run::process_output("/system/bin/mkdir", [~"."]);

        assert_eq!(status, 255);
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
            in_fd: Some(pipe_in.input),
            out_fd: Some(pipe_out.out),
            err_fd: Some(pipe_err.out)
        });

        os::close(pipe_in.input);
        os::close(pipe_out.out);
        os::close(pipe_err.out);

        let expected = ~"test";
        writeclose(pipe_in.out, expected);
        let actual = readclose(pipe_out.input);
        readclose(pipe_err.input);
        proc.finish();

        assert_eq!(expected, actual);
    }

    fn writeclose(fd: c_int, s: &str) {
        let mut writer = file::FileDesc::new(fd);
        writer.write(s.as_bytes());
    }

    fn readclose(fd: c_int) -> ~str {
        let mut res = ~[];
        let mut reader = file::FileDesc::new(fd);
        let mut buf = [0, ..1024];
        loop {
            match reader.read(buf) {
                Some(n) => { res.push_all(buf.slice_to(n)); }
                None => break
            }
        }
        str::from_utf8_owned(res)
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_finish_once() {
        let mut prog = run::Process::new("false", [], run::ProcessOptions::new());
        assert_eq!(prog.finish(), 1);
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_finish_once() {
        let mut prog = run::Process::new("/system/bin/sh", [~"-c",~"false"],
                                         run::ProcessOptions::new());
        assert_eq!(prog.finish(), 1);
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_finish_twice() {
        let mut prog = run::Process::new("false", [], run::ProcessOptions::new());
        assert_eq!(prog.finish(), 1);
        assert_eq!(prog.finish(), 1);
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_finish_twice() {
        let mut prog = run::Process::new("/system/bin/sh", [~"-c",~"false"],
                                         run::ProcessOptions::new());
        assert_eq!(prog.finish(), 1);
        assert_eq!(prog.finish(), 1);
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_finish_with_output_once() {

        let mut prog = run::Process::new("echo", [~"hello"], run::ProcessOptions::new());
        let run::ProcessOutput {status, output, error}
            = prog.finish_with_output();
        let output_str = str::from_utf8(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, ~[]);
        }
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_finish_with_output_once() {

        let mut prog = run::Process::new("/system/bin/sh", [~"-c",~"echo hello"],
                                         run::ProcessOptions::new());
        let run::ProcessOutput {status, output, error}
            = prog.finish_with_output();
        let output_str = str::from_utf8(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, ~[]);
        }
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_finish_with_output_twice() {

        let mut prog = run::Process::new("echo", [~"hello"], run::ProcessOptions::new());
        let run::ProcessOutput {status, output, error}
            = prog.finish_with_output();

        let output_str = str::from_utf8(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, ~[]);
        }

        let run::ProcessOutput {status, output, error}
            = prog.finish_with_output();

        assert_eq!(status, 0);
        assert_eq!(output, ~[]);
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, ~[]);
        }
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_finish_with_output_twice() {

        let mut prog = run::Process::new("/system/bin/sh", [~"-c",~"echo hello"],
                                         run::ProcessOptions::new());
        let run::ProcessOutput {status, output, error}
            = prog.finish_with_output();

        let output_str = str::from_utf8(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, ~[]);
        }

        let run::ProcessOutput {status, output, error}
            = prog.finish_with_output();

        assert_eq!(status, 0);
        assert_eq!(output, ~[]);
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, ~[]);
        }
    }

    #[cfg(unix,not(target_os="android"))]
    fn run_pwd(dir: Option<&Path>) -> run::Process {
        run::Process::new("pwd", [], run::ProcessOptions {
            dir: dir,
            .. run::ProcessOptions::new()
        })
    }
    #[cfg(unix,target_os="android")]
    fn run_pwd(dir: Option<&Path>) -> run::Process {
        run::Process::new("/system/bin/sh", [~"-c",~"pwd"], run::ProcessOptions {
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

        let output = str::from_utf8(prog.finish_with_output().output);
        let parent_dir = os::getcwd();
        let child_dir = Path::new(output.trim());

        let parent_stat = parent_dir.stat().unwrap();
        let child_stat = child_dir.stat().unwrap();

        assert_eq!(parent_stat.device, child_stat.device);
        assert_eq!(parent_stat.inode, child_stat.inode);
    }

    #[test]
    fn test_change_working_directory() {
        // test changing to the parent of os::getcwd() because we know
        // the path exists (and os::getcwd() is not expected to be root)
        let parent_dir = os::getcwd().dir_path();
        let mut prog = run_pwd(Some(&parent_dir));

        let output = str::from_utf8(prog.finish_with_output().output);
        let child_dir = Path::new(output.trim());

        let parent_stat = parent_dir.stat().unwrap();
        let child_stat = child_dir.stat().unwrap();

        assert_eq!(parent_stat.device, child_stat.device);
        assert_eq!(parent_stat.inode, child_stat.inode);
    }

    #[cfg(unix,not(target_os="android"))]
    fn run_env(env: Option<~[(~str, ~str)]>) -> run::Process {
        run::Process::new("env", [], run::ProcessOptions {
            env: env,
            .. run::ProcessOptions::new()
        })
    }
    #[cfg(unix,target_os="android")]
    fn run_env(env: Option<~[(~str, ~str)]>) -> run::Process {
        run::Process::new("/system/bin/sh", [~"-c",~"set"], run::ProcessOptions {
            env: env,
            .. run::ProcessOptions::new()
        })
    }

    #[cfg(windows)]
    fn run_env(env: Option<~[(~str, ~str)]>) -> run::Process {
        run::Process::new("cmd", [~"/c", ~"set"], run::ProcessOptions {
            env: env,
            .. run::ProcessOptions::new()
        })
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_inherit_env() {
        if running_on_valgrind() { return; }

        let mut prog = run_env(None);
        let output = str::from_utf8(prog.finish_with_output().output);

        let r = os::env();
        for &(ref k, ref v) in r.iter() {
            // don't check windows magical empty-named variables
            assert!(k.is_empty() || output.contains(format!("{}={}", *k, *v)));
        }
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_inherit_env() {
        if running_on_valgrind() { return; }

        let mut prog = run_env(None);
        let output = str::from_utf8(prog.finish_with_output().output);

        let r = os::env();
        for &(ref k, ref v) in r.iter() {
            // don't check android RANDOM variables
            if *k != ~"RANDOM" {
                assert!(output.contains(format!("{}={}", *k, *v)) ||
                        output.contains(format!("{}=\'{}\'", *k, *v)));
            }
        }
    }

    #[test]
    fn test_add_to_env() {

        let mut new_env = os::env();
        new_env.push((~"RUN_TEST_NEW_ENV", ~"123"));

        let mut prog = run_env(Some(new_env));
        let output = str::from_utf8(prog.finish_with_output().output);

        assert!(output.contains("RUN_TEST_NEW_ENV=123"));
    }
}
