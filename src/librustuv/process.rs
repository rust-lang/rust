// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::IoError;
use std::io::process;
use libc::c_int;
use libc;
use std::ptr;
use std::rt::rtio::RtioProcess;
use std::rt::task::BlockedTask;
use std::slice;

use homing::{HomingIO, HomeHandle};
use pipe::PipeWatcher;
use super::{UvHandle, UvError, uv_error_to_io_error,
            wait_until_woken_after, wakeup};
use uvio::UvIoFactory;
use uvll;

pub struct Process {
    handle: *uvll::uv_process_t,
    home: HomeHandle,

    /// Task to wake up (may be null) for when the process exits
    to_wake: Option<BlockedTask>,

    /// Collected from the exit_cb
    exit_status: Option<process::ProcessExit>,
}

impl Process {
    /// Spawn a new process inside the specified event loop.
    ///
    /// Returns either the corresponding process object or an error which
    /// occurred.
    pub fn spawn(io_loop: &mut UvIoFactory, config: process::ProcessConfig)
                -> Result<(~Process, ~[Option<PipeWatcher>]), UvError>
    {
        let cwd = config.cwd.map(|s| s.to_c_str());
        let mut io = ~[config.stdin, config.stdout, config.stderr];
        for slot in config.extra_io.iter() {
            io.push(*slot);
        }
        let mut stdio = slice::with_capacity::<uvll::uv_stdio_container_t>(io.len());
        let mut ret_io = slice::with_capacity(io.len());
        unsafe {
            stdio.set_len(io.len());
            for (slot, other) in stdio.iter().zip(io.iter()) {
                let io = set_stdio(slot as *uvll::uv_stdio_container_t, other,
                                   io_loop);
                ret_io.push(io);
            }
        }

        let ret = with_argv(config.program, config.args, |argv| {
            with_env(config.env, |envp| {
                let mut flags = 0;
                if config.uid.is_some() {
                    flags |= uvll::PROCESS_SETUID;
                }
                if config.gid.is_some() {
                    flags |= uvll::PROCESS_SETGID;
                }
                if config.detach {
                    flags |= uvll::PROCESS_DETACHED;
                }
                let options = uvll::uv_process_options_t {
                    exit_cb: on_exit,
                    file: unsafe { *argv },
                    args: argv,
                    env: envp,
                    cwd: match cwd {
                        Some(ref cwd) => cwd.with_ref(|p| p),
                        None => ptr::null(),
                    },
                    flags: flags as libc::c_uint,
                    stdio_count: stdio.len() as libc::c_int,
                    stdio: stdio.as_ptr(),
                    uid: config.uid.unwrap_or(0) as uvll::uv_uid_t,
                    gid: config.gid.unwrap_or(0) as uvll::uv_gid_t,
                };

                let handle = UvHandle::alloc(None::<Process>, uvll::UV_PROCESS);
                let process = ~Process {
                    handle: handle,
                    home: io_loop.make_handle(),
                    to_wake: None,
                    exit_status: None,
                };
                match unsafe {
                    uvll::uv_spawn(io_loop.uv_loop(), handle, &options)
                } {
                    0 => Ok(process.install()),
                    err => Err(UvError(err)),
                }
            })
        });

        match ret {
            Ok(p) => Ok((p, ret_io)),
            Err(e) => Err(e),
        }
    }

    pub fn kill(pid: libc::pid_t, signum: int) -> Result<(), UvError> {
        match unsafe {
            uvll::uv_kill(pid as libc::c_int, signum as libc::c_int)
        } {
            0 => Ok(()),
            n => Err(UvError(n))
        }
    }
}

extern fn on_exit(handle: *uvll::uv_process_t,
                  exit_status: i64,
                  term_signal: libc::c_int) {
    let p: &mut Process = unsafe { UvHandle::from_uv_handle(&handle) };

    assert!(p.exit_status.is_none());
    p.exit_status = Some(match term_signal {
        0 => process::ExitStatus(exit_status as int),
        n => process::ExitSignal(n as int),
    });

    if p.to_wake.is_none() { return }
    wakeup(&mut p.to_wake);
}

unsafe fn set_stdio(dst: *uvll::uv_stdio_container_t,
                    io: &process::StdioContainer,
                    io_loop: &mut UvIoFactory) -> Option<PipeWatcher> {
    match *io {
        process::Ignored => {
            uvll::set_stdio_container_flags(dst, uvll::STDIO_IGNORE);
            None
        }
        process::InheritFd(fd) => {
            uvll::set_stdio_container_flags(dst, uvll::STDIO_INHERIT_FD);
            uvll::set_stdio_container_fd(dst, fd);
            None
        }
        process::CreatePipe(readable, writable) => {
            let mut flags = uvll::STDIO_CREATE_PIPE as libc::c_int;
            if readable {
                flags |= uvll::STDIO_READABLE_PIPE as libc::c_int;
            }
            if writable {
                flags |= uvll::STDIO_WRITABLE_PIPE as libc::c_int;
            }
            let pipe = PipeWatcher::new(io_loop, false);
            uvll::set_stdio_container_flags(dst, flags);
            uvll::set_stdio_container_stream(dst, pipe.handle());
            Some(pipe)
        }
    }
}

/// Converts the program and arguments to the argv array expected by libuv
fn with_argv<T>(prog: &str, args: &[~str], f: |**libc::c_char| -> T) -> T {
    // First, allocation space to put all the C-strings (we need to have
    // ownership of them somewhere
    let mut c_strs = slice::with_capacity(args.len() + 1);
    c_strs.push(prog.to_c_str());
    for arg in args.iter() {
        c_strs.push(arg.to_c_str());
    }

    // Next, create the char** array
    let mut c_args = slice::with_capacity(c_strs.len() + 1);
    for s in c_strs.iter() {
        c_args.push(s.with_ref(|p| p));
    }
    c_args.push(ptr::null());
    f(c_args.as_ptr())
}

/// Converts the environment to the env array expected by libuv
fn with_env<T>(env: Option<&[(~str, ~str)]>, f: |**libc::c_char| -> T) -> T {
    let env = match env {
        Some(s) => s,
        None => { return f(ptr::null()); }
    };
    // As with argv, create some temporary storage and then the actual array
    let mut envp = slice::with_capacity(env.len());
    for &(ref key, ref value) in env.iter() {
        envp.push(format!("{}={}", *key, *value).to_c_str());
    }
    let mut c_envp = slice::with_capacity(envp.len() + 1);
    for s in envp.iter() {
        c_envp.push(s.with_ref(|p| p));
    }
    c_envp.push(ptr::null());
    f(c_envp.as_ptr())
}

impl HomingIO for Process {
    fn home<'r>(&'r mut self) -> &'r mut HomeHandle { &mut self.home }
}

impl UvHandle<uvll::uv_process_t> for Process {
    fn uv_handle(&self) -> *uvll::uv_process_t { self.handle }
}

impl RtioProcess for Process {
    fn id(&self) -> libc::pid_t {
        unsafe { uvll::process_pid(self.handle) as libc::pid_t }
    }

    fn kill(&mut self, signal: int) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        match unsafe {
            uvll::uv_process_kill(self.handle, signal as libc::c_int)
        } {
            0 => Ok(()),
            err => Err(uv_error_to_io_error(UvError(err)))
        }
    }

    fn wait(&mut self) -> process::ProcessExit {
        // Make sure (on the home scheduler) that we have an exit status listed
        let _m = self.fire_homing_missile();
        match self.exit_status {
            Some(..) => {}
            None => {
                // If there's no exit code previously listed, then the
                // process's exit callback has yet to be invoked. We just
                // need to deschedule ourselves and wait to be reawoken.
                wait_until_woken_after(&mut self.to_wake, &self.uv_loop(), || {});
                assert!(self.exit_status.is_some());
            }
        }

        self.exit_status.unwrap()
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _m = self.fire_homing_missile();
        assert!(self.to_wake.is_none());
        self.close();
    }
}
