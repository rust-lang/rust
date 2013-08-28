// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use libc;
use ptr;
use vec;
use cell::Cell;

use rt::uv;
use rt::uv::net;
use rt::uv::pipe;
use rt::uv::uvll;

/// A process wraps the handle of the underlying uv_process_t.
pub struct Process(*uvll::uv_process_t);

/// This configuration describes how a new process should be spawned. This is
/// translated to libuv's own configuration
pub struct Config<'self> {
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
    io: &'self [StdioContainer]
}

/// Describes what to do with a standard io stream for a child process.
pub enum StdioContainer {
    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`
    Ignored,

    /// The specified file descriptor is inherited for the stream which it is
    /// specified for.
    InheritFd(libc::c_int),

    /// The specified libuv stream is inherited for the corresponding file
    /// descriptor it is assigned to.
    InheritStream(net::StreamWatcher),

    /// Creates a pipe for the specified file descriptor which will be directed
    /// into the previously-initialized pipe passed in.
    ///
    /// The first boolean argument is whether the pipe is readable, and the
    /// second is whether it is writable. These properties are from the view of
    /// the *child* process, not the parent process.
    CreatePipe(pipe::Pipe, bool /* readable */, bool /* writable */),
}

impl uv::Watcher for Process {}

impl Process {
    /// Creates a new process, ready to spawn inside an event loop
    pub fn new() -> Process {
        let handle = unsafe { uvll::malloc_handle(uvll::UV_PROCESS) };
        assert!(handle.is_not_null());
        let mut ret: Process = uv::NativeHandle::from_native_handle(handle);
        ret.install_watcher_data();
        return ret;
    }

    /// Spawn a new process inside the specified event loop.
    ///
    /// The `config` variable will be passed down to libuv, and the `exit_cb`
    /// will be run only once, when the process exits.
    ///
    /// Returns either the corresponding process object or an error which
    /// occurred.
    pub fn spawn(&mut self, loop_: &uv::Loop, config: &Config,
                 exit_cb: uv::ExitCallback) -> Result<(), uv::UvError> {
        let cwd = config.cwd.map_move(|s| s.to_c_str());

        extern fn on_exit(p: *uvll::uv_process_t,
                          exit_status: libc::c_int,
                          term_signal: libc::c_int) {
            let mut p: Process = uv::NativeHandle::from_native_handle(p);
            let err = match exit_status {
                0 => None,
                _ => uv::status_to_maybe_uv_error(-1)
            };
            p.get_watcher_data().exit_cb.take_unwrap()(p,
                                                       exit_status as int,
                                                       term_signal as int,
                                                       err);
        }

        let mut stdio = vec::with_capacity::<uvll::uv_stdio_container_t>(
                                config.io.len());
        unsafe {
            vec::raw::set_len(&mut stdio, config.io.len());
            for (slot, &other) in stdio.iter().zip(config.io.iter()) {
                set_stdio(slot as *uvll::uv_stdio_container_t, other);
            }
        }

        let exit_cb = Cell::new(exit_cb);
        do with_argv(config.program, config.args) |argv| {
            do with_env(config.env) |envp| {
                let options = uvll::uv_process_options_t {
                    exit_cb: on_exit,
                    file: unsafe { *argv },
                    args: argv,
                    env: envp,
                    cwd: match cwd {
                        Some(ref cwd) => cwd.with_ref(|p| p),
                        None => ptr::null(),
                    },
                    flags: 0,
                    stdio_count: stdio.len() as libc::c_int,
                    stdio: stdio.as_imm_buf(|p, _| p),
                    uid: 0,
                    gid: 0,
                };

                match unsafe {
                    uvll::spawn(loop_.native_handle(), **self, options)
                } {
                    0 => {
                        (*self).get_watcher_data().exit_cb = Some(exit_cb.take());
                        Ok(())
                    }
                    err => Err(uv::UvError(err))
                }
            }
        }
    }

    /// Sends a signal to this process.
    ///
    /// This is a wrapper around `uv_process_kill`
    pub fn kill(&self, signum: int) -> Result<(), uv::UvError> {
        match unsafe {
            uvll::process_kill(self.native_handle(), signum as libc::c_int)
        } {
            0 => Ok(()),
            err => Err(uv::UvError(err))
        }
    }

    /// Returns the process id of a spawned process
    pub fn pid(&self) -> libc::pid_t {
        unsafe { uvll::process_pid(**self) as libc::pid_t }
    }

    /// Closes this handle, invoking the specified callback once closed
    pub fn close(self, cb: uv::NullCallback) {
        {
            let mut this = self;
            let data = this.get_watcher_data();
            assert!(data.close_cb.is_none());
            data.close_cb = Some(cb);
        }

        unsafe { uvll::close(self.native_handle(), close_cb); }

        extern fn close_cb(handle: *uvll::uv_process_t) {
            let mut process: Process = uv::NativeHandle::from_native_handle(handle);
            process.get_watcher_data().close_cb.take_unwrap()();
            process.drop_watcher_data();
            unsafe { uvll::free_handle(handle as *libc::c_void) }
        }
    }
}

unsafe fn set_stdio(dst: *uvll::uv_stdio_container_t, io: StdioContainer) {
    match io {
        Ignored => { uvll::set_stdio_container_flags(dst, uvll::STDIO_IGNORE); }
        InheritFd(fd) => {
            uvll::set_stdio_container_flags(dst, uvll::STDIO_INHERIT_FD);
            uvll::set_stdio_container_fd(dst, fd);
        }
        InheritStream(stream) => {
            uvll::set_stdio_container_flags(dst, uvll::STDIO_INHERIT_STREAM);
            uvll::set_stdio_container_stream(dst, stream.native_handle());
        }
        CreatePipe(pipe, readable, writable) => {
            let mut flags = uvll::STDIO_CREATE_PIPE as libc::c_int;
            if readable {
                flags |= uvll::STDIO_READABLE_PIPE as libc::c_int;
            }
            if writable {
                flags |= uvll::STDIO_WRITABLE_PIPE as libc::c_int;
            }
            uvll::set_stdio_container_flags(dst, flags);
            uvll::set_stdio_container_stream(dst,
                                             pipe.as_stream().native_handle());
        }
    }
}

/// Converts the program and arguments to the argv array expected by libuv
fn with_argv<T>(prog: &str, args: &[~str], f: &fn(**libc::c_char) -> T) -> T {
    // First, allocation space to put all the C-strings (we need to have
    // ownership of them somewhere
    let mut c_strs = vec::with_capacity(args.len() + 1);
    c_strs.push(prog.to_c_str());
    for arg in args.iter() {
        c_strs.push(arg.to_c_str());
    }

    // Next, create the char** array
    let mut c_args = vec::with_capacity(c_strs.len() + 1);
    for s in c_strs.iter() {
        c_args.push(s.with_ref(|p| p));
    }
    c_args.push(ptr::null());
    c_args.as_imm_buf(|buf, _| f(buf))
}

/// Converts the environment to the env array expected by libuv
fn with_env<T>(env: Option<&[(~str, ~str)]>, f: &fn(**libc::c_char) -> T) -> T {
    let env = match env {
        Some(s) => s,
        None => { return f(ptr::null()); }
    };
    // As with argv, create some temporary storage and then the actual array
    let mut envp = vec::with_capacity(env.len());
    for &(ref key, ref value) in env.iter() {
        envp.push(fmt!("%s=%s", *key, *value).to_c_str());
    }
    let mut c_envp = vec::with_capacity(envp.len() + 1);
    for s in envp.iter() {
        c_envp.push(s.with_ref(|p| p));
    }
    c_envp.push(ptr::null());
    c_envp.as_imm_buf(|buf, _| f(buf))
}

impl uv::NativeHandle<*uvll::uv_process_t> for Process {
    fn from_native_handle(handle: *uvll::uv_process_t) -> Process {
        Process(handle)
    }
    fn native_handle(&self) -> *uvll::uv_process_t {
        match self { &Process(ptr) => ptr }
    }
}
