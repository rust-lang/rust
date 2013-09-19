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
use cell::Cell;
use libc;
use ptr;
use util;
use vec;

use rt::io::process::*;
use rt::uv;
use rt::uv::uvio::UvPipeStream;
use rt::uv::uvll;

/// A process wraps the handle of the underlying uv_process_t.
pub struct Process(*uvll::uv_process_t);

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
    pub fn spawn(&mut self, loop_: &uv::Loop, mut config: ProcessConfig,
                 exit_cb: uv::ExitCallback)
                    -> Result<~[Option<UvPipeStream>], uv::UvError>
    {
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

        let io = util::replace(&mut config.io, ~[]);
        let mut stdio = vec::with_capacity::<uvll::uv_stdio_container_t>(io.len());
        let mut ret_io = vec::with_capacity(io.len());
        unsafe {
            vec::raw::set_len(&mut stdio, io.len());
            for (slot, other) in stdio.iter().zip(io.move_iter()) {
                let io = set_stdio(slot as *uvll::uv_stdio_container_t, other);
                ret_io.push(io);
            }
        }

        let exit_cb = Cell::new(exit_cb);
        let ret_io = Cell::new(ret_io);
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
                        Ok(ret_io.take())
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

unsafe fn set_stdio(dst: *uvll::uv_stdio_container_t,
                    io: StdioContainer) -> Option<UvPipeStream> {
    match io {
        Ignored => {
            uvll::set_stdio_container_flags(dst, uvll::STDIO_IGNORE);
            None
        }
        InheritFd(fd) => {
            uvll::set_stdio_container_flags(dst, uvll::STDIO_INHERIT_FD);
            uvll::set_stdio_container_fd(dst, fd);
            None
        }
        CreatePipe(pipe, readable, writable) => {
            let mut flags = uvll::STDIO_CREATE_PIPE as libc::c_int;
            if readable {
                flags |= uvll::STDIO_READABLE_PIPE as libc::c_int;
            }
            if writable {
                flags |= uvll::STDIO_WRITABLE_PIPE as libc::c_int;
            }
            let handle = pipe.pipe.as_stream().native_handle();
            uvll::set_stdio_container_flags(dst, flags);
            uvll::set_stdio_container_stream(dst, handle);
            Some(pipe.bind())
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
