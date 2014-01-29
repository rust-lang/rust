// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Bindings for executing child processes

use prelude::*;

use libc;
use io;
use io::io_error;
use rt::rtio::{RtioProcess, IoFactory, LocalIo};

use fmt;

// windows values don't matter as long as they're at least one of unix's
// TERM/KILL/INT signals
#[cfg(windows)] pub static PleaseExitSignal: int = 15;
#[cfg(windows)] pub static MustDieSignal: int = 9;
#[cfg(not(windows))] pub static PleaseExitSignal: int = libc::SIGTERM as int;
#[cfg(not(windows))] pub static MustDieSignal: int = libc::SIGKILL as int;

pub struct Process {
    priv handle: ~RtioProcess,
    io: ~[Option<io::PipeStream>],
}

/// This configuration describes how a new process should be spawned. This is
/// translated to libuv's own configuration
pub struct ProcessConfig<'a> {
    /// Path to the program to run
    program: &'a str,

    /// Arguments to pass to the program (doesn't include the program itself)
    args: &'a [~str],

    /// Optional environment to specify for the program. If this is None, then
    /// it will inherit the current process's environment.
    env: Option<&'a [(~str, ~str)]>,

    /// Optional working directory for the new process. If this is None, then
    /// the current directory of the running process is inherited.
    cwd: Option<&'a str>,

    /// Any number of streams/file descriptors/pipes may be attached to this
    /// process. This list enumerates the file descriptors and such for the
    /// process to be spawned, and the file descriptors inherited will start at
    /// 0 and go to the length of this array.
    ///
    /// Standard file descriptors are:
    ///
    ///     0 - stdin
    ///     1 - stdout
    ///     2 - stderr
    io: &'a [StdioContainer]
}

/// Describes what to do with a standard io stream for a child process.
pub enum StdioContainer {
    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`
    Ignored,

    /// The specified file descriptor is inherited for the stream which it is
    /// specified for.
    InheritFd(libc::c_int),

    /// Creates a pipe for the specified file descriptor which will be created
    /// when the process is spawned.
    ///
    /// The first boolean argument is whether the pipe is readable, and the
    /// second is whether it is writable. These properties are from the view of
    /// the *child* process, not the parent process.
    CreatePipe(bool /* readable */, bool /* writable */),
}

/// Describes the result of a process after it has terminated.
/// Note that Windows have no signals, so the result is usually ExitStatus.
#[deriving(Eq)]
pub enum ProcessExit {
    /// Normal termination with an exit status.
    ExitStatus(int),

    /// Termination by signal, with the signal number.
    ExitSignal(int),
}

impl fmt::Show for ProcessExit {
    /// Format a ProcessExit enum, to nicely present the information.
    fn fmt(obj: &ProcessExit, f: &mut fmt::Formatter) {
        match *obj {
            ExitStatus(code) =>  write!(f.buf, "exit code: {}", code),
            ExitSignal(code) =>  write!(f.buf, "signal: {}", code),
        }
    }
}

impl ProcessExit {
    /// Was termination successful? Signal termination not considered a success,
    /// and success is defined as a zero exit status.
    pub fn success(&self) -> bool {
        return self.matches_exit_status(0);
    }

    /// Checks whether this ProcessExit matches the given exit status.
    /// Termination by signal will never match an exit code.
    pub fn matches_exit_status(&self, wanted: int) -> bool {
        *self == ExitStatus(wanted)
    }
}

impl Process {
    /// Creates a new pipe initialized, but not bound to any particular
    /// source/destination
    pub fn new(config: ProcessConfig) -> Option<Process> {
        let mut config = Some(config);
        LocalIo::maybe_raise(|io| {
            io.spawn(config.take_unwrap()).map(|(p, io)| {
                Process {
                    handle: p,
                    io: io.move_iter().map(|p| {
                        p.map(|p| io::PipeStream::new(p))
                    }).collect()
                }
            })
        })
    }

    /// Returns the process id of this child process
    pub fn id(&self) -> libc::pid_t { self.handle.id() }

    /// Sends the specified signal to the child process, returning whether the
    /// signal could be delivered or not.
    ///
    /// Note that this is purely a wrapper around libuv's `uv_process_kill`
    /// function.
    ///
    /// If the signal delivery fails, then the `io_error` condition is raised on
    pub fn signal(&mut self, signal: int) {
        match self.handle.kill(signal) {
            Ok(()) => {}
            Err(err) => {
                io_error::cond.raise(err)
            }
        }
    }

    /// Wait for the child to exit completely, returning the status that it
    /// exited with. This function will continue to have the same return value
    /// after it has been called at least once.
    pub fn wait(&mut self) -> ProcessExit { self.handle.wait() }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Close all I/O before exiting to ensure that the child doesn't wait
        // forever to print some text or something similar.
        loop {
            match self.io.pop() {
                Some(_) => (),
                None => break,
            }
        }

        self.wait();
    }
}

#[cfg(test)]
mod tests {
    use io::process::{ProcessConfig, Process};
    use prelude::*;
    use str;

    // FIXME(#10380)
    #[cfg(unix, not(target_os="android"))]
    iotest!(fn smoke() {
        let io = ~[];
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &[~"-c", ~"true"],
            env: None,
            cwd: None,
            io: io,
        };
        let p = Process::new(args);
        assert!(p.is_some());
        let mut p = p.unwrap();
        assert!(p.wait().success());
    })

    // FIXME(#10380)
    #[cfg(unix, not(target_os="android"))]
    iotest!(fn smoke_failure() {
        let io = ~[];
        let args = ProcessConfig {
            program: "if-this-is-a-binary-then-the-world-has-ended",
            args: &[],
            env: None,
            cwd: None,
            io: io,
        };
        match io::result(|| Process::new(args)) {
            Ok(..) => fail!(),
            Err(..) => {}
        }
    })

    // FIXME(#10380)
    #[cfg(unix, not(target_os="android"))]
    iotest!(fn exit_reported_right() {
        let io = ~[];
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &[~"-c", ~"exit 1"],
            env: None,
            cwd: None,
            io: io,
        };
        let p = Process::new(args);
        assert!(p.is_some());
        let mut p = p.unwrap();
        assert!(p.wait().matches_exit_status(1));
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn signal_reported_right() {
        let io = ~[];
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &[~"-c", ~"kill -1 $$"],
            env: None,
            cwd: None,
            io: io,
        };
        let p = Process::new(args);
        assert!(p.is_some());
        let mut p = p.unwrap();
        match p.wait() {
            process::ExitSignal(1) => {},
            result => fail!("not terminated by signal 1 (instead, {})", result),
        }
    })

    pub fn read_all(input: &mut Reader) -> ~str {
        let mut ret = ~"";
        let mut buf = [0, ..1024];
        loop {
            match input.read(buf) {
                None => { break }
                Some(n) => { ret.push_str(str::from_utf8(buf.slice_to(n)).unwrap()); }
            }
        }
        return ret;
    }

    pub fn run_output(args: ProcessConfig) -> ~str {
        let p = Process::new(args);
        assert!(p.is_some());
        let mut p = p.unwrap();
        assert!(p.io[0].is_none());
        assert!(p.io[1].is_some());
        let ret = read_all(p.io[1].get_mut_ref() as &mut Reader);
        assert!(p.wait().success());
        return ret;
    }

    // FIXME(#10380)
    #[cfg(unix, not(target_os="android"))]
    iotest!(fn stdout_works() {
        let io = ~[Ignored, CreatePipe(false, true)];
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &[~"-c", ~"echo foobar"],
            env: None,
            cwd: None,
            io: io,
        };
        assert_eq!(run_output(args), ~"foobar\n");
    })

    // FIXME(#10380)
    #[cfg(unix, not(target_os="android"))]
    iotest!(fn set_cwd_works() {
        let io = ~[Ignored, CreatePipe(false, true)];
        let cwd = Some("/");
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &[~"-c", ~"pwd"],
            env: None,
            cwd: cwd,
            io: io,
        };
        assert_eq!(run_output(args), ~"/\n");
    })

    // FIXME(#10380)
    #[cfg(unix, not(target_os="android"))]
    iotest!(fn stdin_works() {
        let io = ~[CreatePipe(true, false),
                   CreatePipe(false, true)];
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &[~"-c", ~"read line; echo $line"],
            env: None,
            cwd: None,
            io: io,
        };
        let mut p = Process::new(args).expect("didn't create a proces?!");
        p.io[0].get_mut_ref().write("foobar".as_bytes());
        p.io[0] = None; // close stdin;
        let out = read_all(p.io[1].get_mut_ref() as &mut Reader);
        assert!(p.wait().success());
        assert_eq!(out, ~"foobar\n");
    })

}
