// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::c_str::CString;
use std::cast;
use std::libc;
use std::rt::BlockedTask;
use std::rt::io::IoError;
use std::rt::local::Local;
use std::rt::rtio::{RtioPipe, RtioUnixListener, RtioUnixAcceptor};
use std::rt::sched::{Scheduler, SchedHandle};
use std::rt::tube::Tube;

use stream::StreamWatcher;
use super::{Loop, UvError, UvHandle, Request, uv_error_to_io_error};
use uvio::HomingIO;
use uvll;

pub struct PipeWatcher {
    stream: StreamWatcher,
    home: SchedHandle,
    priv defused: bool,
}

pub struct PipeListener {
    home: SchedHandle,
    pipe: *uvll::uv_pipe_t,
    priv closing_task: Option<BlockedTask>,
    priv outgoing: Tube<Result<~RtioPipe, IoError>>,
}

pub struct PipeAcceptor {
    listener: ~PipeListener,
    priv incoming: Tube<Result<~RtioPipe, IoError>>,
}

// PipeWatcher implementation and traits

impl PipeWatcher {
    // Creates an uninitialized pipe watcher. The underlying uv pipe is ready to
    // get bound to some other source (this is normally a helper method paired
    // with another call).
    pub fn new(loop_: &Loop, ipc: bool) -> PipeWatcher {
        let handle = unsafe {
            let handle = uvll::malloc_handle(uvll::UV_NAMED_PIPE);
            assert!(!handle.is_null());
            let ipc = ipc as libc::c_int;
            assert_eq!(uvll::uv_pipe_init(loop_.handle, handle, ipc), 0);
            handle
        };
        PipeWatcher {
            stream: StreamWatcher::new(handle),
            home: get_handle_to_current_scheduler!(),
            defused: false,
        }
    }

    pub fn open(loop_: &Loop, file: libc::c_int) -> Result<PipeWatcher, UvError>
    {
        let pipe = PipeWatcher::new(loop_, false);
        match unsafe { uvll::uv_pipe_open(pipe.handle(), file) } {
            0 => Ok(pipe),
            n => Err(UvError(n))
        }
    }

    pub fn connect(loop_: &Loop, name: &CString) -> Result<PipeWatcher, UvError>
    {
        struct Ctx { task: Option<BlockedTask>, result: libc::c_int, }
        let mut cx = Ctx { task: None, result: 0 };
        let req = Request::new(uvll::UV_CONNECT);
        let pipe = PipeWatcher::new(loop_, false);
        unsafe {
            uvll::set_data_for_req(req.handle, &cx as *Ctx);
            uvll::uv_pipe_connect(req.handle,
                                  pipe.handle(),
                                  name.with_ref(|p| p),
                                  connect_cb)
        }
        req.defuse();

        let sched: ~Scheduler = Local::take();
        do sched.deschedule_running_task_and_then |_, task| {
            cx.task = Some(task);
        }
        return match cx.result {
            0 => Ok(pipe),
            n => Err(UvError(n))
        };

        extern fn connect_cb(req: *uvll::uv_connect_t, status: libc::c_int) {
            let _req = Request::wrap(req);
            if status == uvll::ECANCELED { return }
            unsafe {
                let cx: &mut Ctx = cast::transmute(uvll::get_data_for_req(req));
                cx.result = status;
                let sched: ~Scheduler = Local::take();
                sched.resume_blocked_task_immediately(cx.task.take_unwrap());
            }
        }
    }

    pub fn handle(&self) -> *uvll::uv_pipe_t { self.stream.handle }

    // Unwraps the underlying uv pipe. This cancels destruction of the pipe and
    // allows the pipe to get moved elsewhere
    fn unwrap(mut self) -> *uvll::uv_pipe_t {
        self.defused = true;
        return self.stream.handle;
    }
}

impl RtioPipe for PipeWatcher {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        let _m = self.fire_homing_missile();
        self.stream.read(buf).map_err(uv_error_to_io_error)
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        self.stream.write(buf).map_err(uv_error_to_io_error)
    }
}

impl HomingIO for PipeWatcher {
    fn home<'a>(&'a mut self) -> &'a mut SchedHandle { &mut self.home }
}

impl Drop for PipeWatcher {
    fn drop(&mut self) {
        if !self.defused {
            let _m = self.fire_homing_missile();
            self.stream.close();
        }
    }
}

extern fn pipe_close_cb(handle: *uvll::uv_handle_t) {
    unsafe { uvll::free_handle(handle) }
}

// PipeListener implementation and traits

impl PipeListener {
    pub fn bind(loop_: &Loop, name: &CString) -> Result<~PipeListener, UvError> {
        let pipe = PipeWatcher::new(loop_, false);
        match unsafe { uvll::uv_pipe_bind(pipe.handle(), name.with_ref(|p| p)) } {
            0 => {
                // If successful, unwrap the PipeWatcher because we control how
                // we close the pipe differently. We can't rely on
                // StreamWatcher's default close method.
                let p = ~PipeListener {
                    home: get_handle_to_current_scheduler!(),
                    pipe: pipe.unwrap(),
                    closing_task: None,
                    outgoing: Tube::new(),
                };
                Ok(p.install())
            }
            n => Err(UvError(n))
        }
    }
}

impl RtioUnixListener for PipeListener {
    fn listen(mut ~self) -> Result<~RtioUnixAcceptor, IoError> {
        // create the acceptor object from ourselves
        let incoming = self.outgoing.clone();
        let mut acceptor = ~PipeAcceptor {
            listener: self,
            incoming: incoming,
        };

        let _m = acceptor.fire_homing_missile();
        // XXX: the 128 backlog should be configurable
        match unsafe { uvll::uv_listen(acceptor.listener.pipe, 128, listen_cb) } {
            0 => Ok(acceptor as ~RtioUnixAcceptor),
            n => Err(uv_error_to_io_error(UvError(n))),
        }
    }
}

impl HomingIO for PipeListener {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl UvHandle<uvll::uv_pipe_t> for PipeListener {
    fn uv_handle(&self) -> *uvll::uv_pipe_t { self.pipe }
}

extern fn listen_cb(server: *uvll::uv_stream_t, status: libc::c_int) {
    let msg = match status {
        0 => {
            let loop_ = Loop::wrap(unsafe {
                uvll::get_loop_for_uv_handle(server)
            });
            let client = PipeWatcher::new(&loop_, false);
            assert_eq!(unsafe { uvll::uv_accept(server, client.handle()) }, 0);
            Ok(~client as ~RtioPipe)
        }
        uvll::ECANCELED => return,
        n => Err(uv_error_to_io_error(UvError(n)))
    };

    let pipe: &mut PipeListener = unsafe { UvHandle::from_uv_handle(&server) };
    pipe.outgoing.send(msg);
}

impl Drop for PipeListener {
    fn drop(&mut self) {
        let (_m, sched) = self.fire_homing_missile_sched();

        do sched.deschedule_running_task_and_then |_, task| {
            self.closing_task = Some(task);
            unsafe { uvll::uv_close(self.pipe, listener_close_cb) }
        }
    }
}

extern fn listener_close_cb(handle: *uvll::uv_handle_t) {
    let pipe: &mut PipeListener = unsafe { UvHandle::from_uv_handle(&handle) };
    unsafe { uvll::free_handle(handle) }

    let sched: ~Scheduler = Local::take();
    sched.resume_blocked_task_immediately(pipe.closing_task.take_unwrap());
}

// PipeAcceptor implementation and traits

impl RtioUnixAcceptor for PipeAcceptor {
    fn accept(&mut self) -> Result<~RtioPipe, IoError> {
        let _m = self.fire_homing_missile();
        self.incoming.recv()
    }
}

impl HomingIO for PipeAcceptor {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { self.listener.home() }
}
