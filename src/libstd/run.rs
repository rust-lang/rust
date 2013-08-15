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

use cast;
use cell::Cell;
use comm::{stream, SharedChan, GenericChan, GenericPort};
#[cfg(not(windows))]
use libc;
use libc::{pid_t, c_int};
use prelude::*;
use task;
use vec::ImmutableVector;

use rt::io;
use rt::local::Local;
use rt::rtio::{IoFactoryObject, RtioProcessObject, RtioProcess, IoFactory};
use rt::uv::process;

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

    /// The internal handle to the underlying libuv process.
    priv handle: ~RtioProcessObject,

    /// Some(fd), or None when stdin is being redirected from a fd not created
    /// by Process::new.
    priv input: Option<~io::Writer>,

    /// Some(file), or None when stdout is being redirected to a fd not created
    /// by Process::new.
    priv output: Option<~io::Reader>,

    /// Some(file), or None when stderr is being redirected to a fd not created
    /// by Process::new.
    priv error: Option<~io::Reader>,
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
     * If this is Some(file-descriptor) then the new process will write its
     * output to the given file descriptor, Process.output_redirected() will
     * return true, and Process.output() will fail.
     */
    out_fd: Option<c_int>,

    /**
     * If this is None then a new pipe will be created for the new progam's
     * error stream and Process.error() will provide a Reader to read from this
     * pipe.
     *
     * If this is Some(file-descriptor) then the new process will write its
     * error output to the given file descriptor, Process.error_redirected()
     * will return true, and and Process.error() will fail.
     */
    err_fd: Option<c_int>,
}

impl<'self> ProcessOptions<'self> {
    /// Return a ProcessOptions that has None in every field.
    pub fn new() -> ProcessOptions {
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
    pub fn new(prog: &str, args: &[~str],
               options: ProcessOptions) -> Option<Process> {
        // First, translate all the stdio options into their libuv equivalents
        let (uv_stdin, stdin) = match options.in_fd {
            Some(fd) => (process::InheritFd(fd), None),
            None => {
                let p = io::pipe::PipeStream::new().expect("need stdin pipe");
                (process::CreatePipe(p.uv_pipe(), true, false),
                 Some(~p as ~io::Writer))
            }
        };
        let (uv_stdout, stdout) = match options.out_fd {
            Some(fd) => (process::InheritFd(fd), None),
            None => {
                let p = io::pipe::PipeStream::new().expect("need stdout pipe");
                (process::CreatePipe(p.uv_pipe(), false, true),
                 Some(~p as ~io::Reader))
            }
        };
        let (uv_stderr, stderr) = match options.err_fd {
            Some(fd) => (process::InheritFd(fd), None),
            None => {
                let p = io::pipe::PipeStream::new().expect("need stderr pipe");
                (process::CreatePipe(p.uv_pipe(), false, true),
                 Some(~p as ~io::Reader))
            }
        };

        // Next, massage our options into the libuv options
        let dir = options.dir.map(|d| d.to_str());
        let dir = dir.map(|d| d.as_slice());
        let config = process::Config {
            program: prog,
            args: args,
            env: options.env.map(|e| e.as_slice()),
            cwd: dir,
            io: [uv_stdin, uv_stdout, uv_stderr],
        };

        // Finally, actually spawn the process
        unsafe {
            let io: *mut IoFactoryObject = Local::unsafe_borrow();
            match (*io).spawn(&config) {
                Ok(handle) => {
                    Some(Process {
                        pid: handle.id(),
                        handle: handle,
                        input: stdin,
                        output: stdout,
                        error: stderr,
                    })
                }
                Err(*) => { None }
            }
        }
    }

    /// Returns the unique id of the process
    pub fn get_id(&self) -> pid_t { self.pid }

    /**
     * Returns a rt::io::Writer that can be used to write to this Process's
     * stdin.
     *
     * Fails if this Process's stdin was redirected to an existing file
     * descriptor.
     */
    pub fn input<'a>(&'a mut self) -> &'a mut io::Writer {
        let ret: &mut io::Writer = *self.input.get_mut_ref();
        return ret;
    }

    /**
     * Returns a rt::io::Reader that can be used to read from this Process's
     * stdout.
     *
     * Fails if this Process's stdout was redirected to an existing file
     * descriptor.
     */
    pub fn output<'a>(&'a mut self) -> &'a mut io::Reader {
        let ret: &mut io::Reader = *self.output.get_mut_ref();
        return ret;
    }

    /**
     * Returns a rt::io::Reader that can be used to read from this Process's
     * stderr.
     *
     * Fails if this Process's stderr was redirected to an existing file
     * descriptor.
     */
    pub fn error<'a>(&'a mut self) -> &'a mut io::Reader {
        let ret: &mut io::Reader = *self.error.get_mut_ref();
        return ret;
    }

    /**
     * Closes the handle to stdin, waits for the child process to terminate, and
     * returns the exit code.
     *
     * If the child has already been finished then the exit code is returned.
     */
    pub fn finish(&mut self) -> int {
        // We're not going to be giving any more input, so close the input by
        // destroying it. Also, if the output is desired, then
        // finish_with_output is called so we discard all the outputs here. Note
        // that the process may not terminate if we don't destroy stdio because
        // it'll be waiting in a write which we'll just never read.
        self.input.take();
        self.output.take();
        self.error.take();

        self.handle.wait()
    }

    /**
     * Closes the handle to stdin, waits for the child process to terminate,
     * and reads and returns all remaining output of stdout and stderr, along
     * with the exit code.
     *
     * If the child has already been finished then the exit code and any
     * remaining unread output of stdout and stderr will be returned.
     *
     * This method will fail if the child process's stdout or stderr streams
     * were redirected to existing file descriptors, or if this method has
     * already been called.
     */
    pub fn finish_with_output(&mut self) -> ProcessOutput {
        // This should probably be a helper method in rt::io
        fn read_everything(input: &mut io::Reader) -> ~[u8] {
            let mut result = ~[];
            let mut buf = [0u8, ..1024];
            loop {
                match input.read(buf) {
                    Some(i) => { result = result + buf.slice_to(i) }
                    None => break
                }
            }
            return result;
        }

        let (p, ch) = stream();
        let ch = SharedChan::new(ch);
        let ch_clone = ch.clone();

        let stderr = Cell::new(self.error.take().unwrap());
        do task::spawn {
            let output = read_everything(stderr.take());
            ch.send((2, output));
        }
        let stdout = Cell::new(self.output.take().unwrap());
        do task::spawn {
            let output = read_everything(stdout.take());
            ch_clone.send((1, output));
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

    /**
     * Terminates the process, giving it a chance to clean itself up if
     * this is supported by the operating system.
     *
     * On Posix OSs SIGTERM will be sent to the process. On Win32
     * TerminateProcess(..) will be called.
     */
    pub fn destroy(&mut self) {
        #[cfg(windows)]      fn sigterm() -> int { 15 }
        #[cfg(not(windows))] fn sigterm() -> int { libc::SIGTERM as int }
        self.handle.kill(sigterm());
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
        #[cfg(windows)]      fn sigkill() -> int { 9 }
        #[cfg(not(windows))] fn sigkill() -> int { libc::SIGKILL as int }
        self.handle.kill(sigkill());
        self.finish();
    }
}

impl Drop for Process {
    fn drop(&self) {
        // FIXME(#4330) Need self by value to get mutability.
        let mut_self: &mut Process = unsafe { cast::transmute(self) };
        mut_self.finish();
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
pub fn process_status(prog: &str, args: &[~str]) -> int {
    let mut prog = Process::new(prog, args, ProcessOptions {
        env: None,
        dir: None,
        in_fd: Some(0),
        out_fd: Some(1),
        err_fd: Some(2)
    }).unwrap();
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
    let mut prog = Process::new(prog, args, ProcessOptions::new()).unwrap();
    prog.finish_with_output()
}

#[cfg(test)]
mod tests {
    use os;
    use path::Path;
    use prelude::*;
    use str;
    use super::*;
    use unstable::running_on_valgrind;

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_process_status() {
        assert_eq!(process_status("false", []), 1);
        assert_eq!(process_status("true", []), 0);
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_process_status() {
        assert_eq!(process_status("/system/bin/sh", [~"-c",~"false"]), 1);
        assert_eq!(process_status("/system/bin/sh", [~"-c",~"true"]), 0);
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_process_output_output() {

        let ProcessOutput {status, output, error}
             = process_output("echo", [~"hello"]);
        let output_str = str::from_bytes(output);

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

        let ProcessOutput {status, output, error}
             = process_output("/system/bin/sh", [~"-c",~"echo hello"]);
        let output_str = str::from_bytes(output);

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

        let ProcessOutput {status, output, error}
             = process_output("mkdir", [~"."]);

        assert_eq!(status, 1);
        assert_eq!(output, ~[]);
        assert!(!error.is_empty());
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_process_output_error() {

        let ProcessOutput {status, output, error}
             = process_output("/system/bin/mkdir", [~"."]);

        assert_eq!(status, 255);
        assert_eq!(output, ~[]);
        assert!(!error.is_empty());
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_finish_once() {
        let mut prog = Process::new("false", [], ProcessOptions::new()).unwrap();
        assert_eq!(prog.finish(), 1);
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_finish_once() {
        let mut prog = Process::new("/system/bin/sh", [~"-c",~"false"],
                                    ProcessOptions::new()).unwrap();
        assert_eq!(prog.finish(), 1);
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_finish_twice() {
        let mut prog = Process::new("false", [], ProcessOptions::new()).unwrap();
        assert_eq!(prog.finish(), 1);
        assert_eq!(prog.finish(), 1);
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_finish_twice() {
        let mut prog = Process::new("/system/bin/sh", [~"-c",~"false"],
                                    ProcessOptions::new()).unwrap();
        assert_eq!(prog.finish(), 1);
        assert_eq!(prog.finish(), 1);
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_finish_with_output_once() {

        let prog = Process::new("echo", [~"hello"], ProcessOptions::new());
        let mut prog = prog.unwrap();
        let ProcessOutput {status, output, error}
            = prog.finish_with_output();
        let output_str = str::from_bytes(output);

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

        let mut prog = Process::new("/system/bin/sh", [~"-c",~"echo hello"],
                                    ProcessOptions::new()).unwrap();
        let ProcessOutput {status, output, error}
            = prog.finish_with_output();
        let output_str = str::from_bytes(output);

        assert_eq!(status, 0);
        assert_eq!(output_str.trim().to_owned(), ~"hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, ~[]);
        }
    }

    #[test]
    #[should_fail]
    #[cfg(not(windows),not(target_os="android"))]
    fn test_finish_with_output_redirected() {
        let mut prog = Process::new("echo", [~"hello"], ProcessOptions {
            env: None,
            dir: None,
            in_fd: Some(0),
            out_fd: Some(1),
            err_fd: Some(2)
        }).unwrap();
        // this should fail because it is not valid to read the output when it
        // was redirected
        prog.finish_with_output();
    }
    #[test]
    #[should_fail]
    #[cfg(not(windows),target_os="android")]
    fn test_finish_with_output_redirected() {
        let mut prog = Process::new("/system/bin/sh", [~"-c",~"echo hello"],
                                    ProcessOptions {
            env: None,
            dir: None,
            in_fd: Some(0),
            out_fd: Some(1),
            err_fd: Some(2)
        }).unwrap();
        // this should fail because it is not valid to read the output when it
        // was redirected
        prog.finish_with_output();
    }

    #[cfg(unix,not(target_os="android"))]
    fn run_pwd(dir: Option<&Path>) -> Process {
        Process::new("pwd", [], ProcessOptions {
            dir: dir,
            .. ProcessOptions::new()
        }).unwrap()
    }
    #[cfg(unix,target_os="android")]
    fn run_pwd(dir: Option<&Path>) -> Process {
        Process::new("/system/bin/sh", [~"-c",~"pwd"], ProcessOptions {
            dir: dir,
            .. ProcessOptions::new()
        }).unwrap()
    }

    #[cfg(windows)]
    fn run_pwd(dir: Option<&Path>) -> Process {
        Process::new("cmd", [~"/c", ~"cd"], ProcessOptions {
            dir: dir,
            .. ProcessOptions::new()
        }).unwrap()
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

    #[cfg(unix,not(target_os="android"))]
    fn run_env(env: Option<~[(~str, ~str)]>) -> Process {
        Process::new("env", [], ProcessOptions {
            env: env,
            .. ProcessOptions::new()
        }).unwrap()
    }
    #[cfg(unix,target_os="android")]
    fn run_env(env: Option<~[(~str, ~str)]>) -> Process {
        Process::new("/system/bin/sh", [~"-c",~"set"], ProcessOptions {
            env: env,
            .. ProcessOptions::new()
        }).unwrap()
    }

    #[cfg(windows)]
    fn run_env(env: Option<~[(~str, ~str)]>) -> Process {
        Process::new("cmd", [~"/c", ~"set"], ProcessOptions {
            env: env,
            .. ProcessOptions::new()
        }).unwrap()
    }

    #[test]
    #[cfg(not(target_os="android"))]
    fn test_inherit_env() {
        if running_on_valgrind() { return; }

        let mut prog = run_env(None);
        let output = str::from_bytes(prog.finish_with_output().output);

        let r = os::env();
        for &(ref k, ref v) in r.iter() {
            // don't check windows magical empty-named variables
            assert!(k.is_empty() || output.contains(fmt!("%s=%s", *k, *v)));
        }
    }
    #[test]
    #[cfg(target_os="android")]
    fn test_inherit_env() {
        if running_on_valgrind() { return; }

        let mut prog = run_env(None);
        let output = str::from_bytes(prog.finish_with_output().output);

        let r = os::env();
        for &(ref k, ref v) in r.iter() {
            // don't check android RANDOM variables
            if *k != ~"RANDOM" {
                assert!(output.contains(fmt!("%s=%s", *k, *v)) ||
                        output.contains(fmt!("%s=\'%s\'", *k, *v)));
            }
        }
    }

    #[test]
    fn test_add_to_env() {
        let mut new_env = os::env();
        new_env.push((~"RUN_TEST_NEW_ENV", ~"123"));

        let mut prog = run_env(Some(new_env));
        let output = str::from_bytes(prog.finish_with_output().output);

        assert!(output.contains("RUN_TEST_NEW_ENV=123"));
    }
}
