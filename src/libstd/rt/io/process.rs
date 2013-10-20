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
use rt::io;
use rt::io::io_error;
use rt::local::Local;
use rt::rtio::{RtioProcess, RtioProcessObject, IoFactoryObject, IoFactory};

// windows values don't matter as long as they're at least one of unix's
// TERM/KILL/INT signals
#[cfg(windows)] pub static PleaseExitSignal: int = 15;
#[cfg(windows)] pub static MustDieSignal: int = 9;
#[cfg(not(windows))] pub static PleaseExitSignal: int = libc::SIGTERM as int;
#[cfg(not(windows))] pub static MustDieSignal: int = libc::SIGKILL as int;

pub struct Process {
    priv handle: ~RtioProcessObject,
    io: ~[Option<io::PipeStream>],
}

/// This configuration describes how a new process should be spawned. This is
/// translated to libuv's own configuration
pub struct ProcessConfig<'self> {
    /// Path to the program to run
    program: &'self str,

    /// Arguments to pass to the program (doesn't include the program itself)
    args: &'self [~str],

    /// Optional environment to specify for the program. If this is None, then
    /// it will inherit the current process's environment.
    env: Option<&'self [(~str, ~str)]>,

    /// Optional working directory for the new process. If this is None, then
    /// the current directory of the running process is inherited.
    cwd: Option<&'self str>,

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
    io: ~[StdioContainer]
}

/// Describes what to do with a standard io stream for a child process.
pub enum StdioContainer {
    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`
    Ignored,

    /// The specified file descriptor is inherited for the stream which it is
    /// specified for.
    InheritFd(libc::c_int),

    // XXX: these two shouldn't have libuv-specific implementation details

    /// The specified libuv stream is inherited for the corresponding file
    /// descriptor it is assigned to.
    // XXX: this needs to be thought out more.
    //InheritStream(uv::net::StreamWatcher),

    /// Creates a pipe for the specified file descriptor which will be directed
    /// into the previously-initialized pipe passed in.
    ///
    /// The first boolean argument is whether the pipe is readable, and the
    /// second is whether it is writable. These properties are from the view of
    /// the *child* process, not the parent process.
    CreatePipe(io::UnboundPipeStream,
               bool /* readable */,
               bool /* writable */),
}

impl Process {
    /// Creates a new pipe initialized, but not bound to any particular
    /// source/destination
    pub fn new(config: ProcessConfig) -> Option<Process> {
        let process = unsafe {
            let io: *mut IoFactoryObject = Local::unsafe_borrow();
            (*io).spawn(config)
        };
        match process {
            Ok((p, io)) => Some(Process{
                handle: p,
                io: io.move_iter().map(|p|
                    p.map(|p| io::PipeStream::bind(p))
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
    pub fn wait(&mut self) -> int { self.handle.wait() }
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
