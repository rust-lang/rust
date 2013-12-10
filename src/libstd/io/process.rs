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

impl fmt::Default for ProcessExit {
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
        let mut io = LocalIo::borrow();
        match io.get().spawn(config) {
            Ok((p, io)) => Some(Process{
                handle: p,
                io: io.move_iter().map(|p|
                    p.map(|p| io::PipeStream::new(p))
                ).collect()
            }),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
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
        for _ in range(0, self.io.len()) {
            self.io.pop();
        }

        self.wait();
    }
}

// Tests for this module can be found in the rtio-processes run-pass test, along
// with the justification for why it's not located here.
