// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use std::c_str::CString;
use std::io::IoError;
use std::rt::rtio::{RtioPipe, RtioUnixListener, RtioUnixAcceptor};

use access::Access;
use homing::{HomingIO, HomeHandle};
use net;
use rc::Refcount;
use stream::StreamWatcher;
use super::{Loop, UvError, UvHandle, uv_error_to_io_error};
use uvio::UvIoFactory;
use uvll;

pub struct PipeWatcher {
    stream: StreamWatcher,
    home: HomeHandle,
    defused: bool,
    refcount: Refcount,

    // see comments in TcpWatcher for why these exist
    write_access: Access,
    read_access: Access,
}

pub struct PipeListener {
    home: HomeHandle,
    pipe: *uvll::uv_pipe_t,
    outgoing: Sender<Result<Box<RtioPipe:Send>, IoError>>,
    incoming: Receiver<Result<Box<RtioPipe:Send>, IoError>>,
}

pub struct PipeAcceptor {
    listener: Box<PipeListener>,
    timeout: net::AcceptTimeout,
}

// PipeWatcher implementation and traits

impl PipeWatcher {
    // Creates an uninitialized pipe watcher. The underlying uv pipe is ready to
    // get bound to some other source (this is normally a helper method paired
    // with another call).
    pub fn new(io: &mut UvIoFactory, ipc: bool) -> PipeWatcher {
        let home = io.make_handle();
        PipeWatcher::new_home(&io.loop_, home, ipc)
    }

    pub fn new_home(loop_: &Loop, home: HomeHandle, ipc: bool) -> PipeWatcher {
        let handle = unsafe {
            let handle = uvll::malloc_handle(uvll::UV_NAMED_PIPE);
            assert!(!handle.is_null());
            let ipc = ipc as libc::c_int;
            assert_eq!(uvll::uv_pipe_init(loop_.handle, handle, ipc), 0);
            handle
        };
        PipeWatcher {
            stream: StreamWatcher::new(handle),
            home: home,
            defused: false,
            refcount: Refcount::new(),
            read_access: Access::new(),
            write_access: Access::new(),
        }
    }

    pub fn open(io: &mut UvIoFactory, file: libc::c_int)
        -> Result<PipeWatcher, UvError>
    {
        let pipe = PipeWatcher::new(io, false);
        match unsafe { uvll::uv_pipe_open(pipe.handle(), file) } {
            0 => Ok(pipe),
            n => Err(UvError(n))
        }
    }

    pub fn connect(io: &mut UvIoFactory, name: &CString, timeout: Option<u64>)
        -> Result<PipeWatcher, UvError>
    {
        let pipe = PipeWatcher::new(io, false);
        let cx = net::ConnectCtx { status: -1, task: None, timer: None };
        cx.connect(pipe, timeout, io, |req, pipe, cb| {
            unsafe {
                uvll::uv_pipe_connect(req.handle, pipe.handle(),
                                      name.with_ref(|p| p), cb)
            }
            0
        })
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
        let m = self.fire_homing_missile();
        let _g = self.read_access.grant(m);
        self.stream.read(buf).map_err(uv_error_to_io_error)
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let m = self.fire_homing_missile();
        let _g = self.write_access.grant(m);
        self.stream.write(buf).map_err(uv_error_to_io_error)
    }

    fn clone(&self) -> Box<RtioPipe:Send> {
        box PipeWatcher {
            stream: StreamWatcher::new(self.stream.handle),
            defused: false,
            home: self.home.clone(),
            refcount: self.refcount.clone(),
            read_access: self.read_access.clone(),
            write_access: self.write_access.clone(),
        } as Box<RtioPipe:Send>
    }
}

impl HomingIO for PipeWatcher {
    fn home<'a>(&'a mut self) -> &'a mut HomeHandle { &mut self.home }
}

impl UvHandle<uvll::uv_pipe_t> for PipeWatcher {
    fn uv_handle(&self) -> *uvll::uv_pipe_t { self.stream.handle }
}

impl Drop for PipeWatcher {
    fn drop(&mut self) {
        let _m = self.fire_homing_missile();
        if !self.defused && self.refcount.decrement() {
            self.close();
        }
    }
}

// PipeListener implementation and traits

impl PipeListener {
    pub fn bind(io: &mut UvIoFactory, name: &CString)
        -> Result<Box<PipeListener>, UvError>
    {
        let pipe = PipeWatcher::new(io, false);
        match unsafe {
            uvll::uv_pipe_bind(pipe.handle(), name.with_ref(|p| p))
        } {
            0 => {
                // If successful, unwrap the PipeWatcher because we control how
                // we close the pipe differently. We can't rely on
                // StreamWatcher's default close method.
                let (tx, rx) = channel();
                let p = box PipeListener {
                    home: io.make_handle(),
                    pipe: pipe.unwrap(),
                    incoming: rx,
                    outgoing: tx,
                };
                Ok(p.install())
            }
            n => Err(UvError(n))
        }
    }
}

impl RtioUnixListener for PipeListener {
    fn listen(~self) -> Result<Box<RtioUnixAcceptor:Send>, IoError> {
        // create the acceptor object from ourselves
        let mut acceptor = box PipeAcceptor {
            listener: self,
            timeout: net::AcceptTimeout::new(),
        };

        let _m = acceptor.fire_homing_missile();
        // FIXME: the 128 backlog should be configurable
        match unsafe { uvll::uv_listen(acceptor.listener.pipe, 128, listen_cb) } {
            0 => Ok(acceptor as Box<RtioUnixAcceptor:Send>),
            n => Err(uv_error_to_io_error(UvError(n))),
        }
    }
}

impl HomingIO for PipeListener {
    fn home<'r>(&'r mut self) -> &'r mut HomeHandle { &mut self.home }
}

impl UvHandle<uvll::uv_pipe_t> for PipeListener {
    fn uv_handle(&self) -> *uvll::uv_pipe_t { self.pipe }
}

extern fn listen_cb(server: *uvll::uv_stream_t, status: libc::c_int) {
    assert!(status != uvll::ECANCELED);

    let pipe: &mut PipeListener = unsafe { UvHandle::from_uv_handle(&server) };
    let msg = match status {
        0 => {
            let loop_ = Loop::wrap(unsafe {
                uvll::get_loop_for_uv_handle(server)
            });
            let client = PipeWatcher::new_home(&loop_, pipe.home().clone(), false);
            assert_eq!(unsafe { uvll::uv_accept(server, client.handle()) }, 0);
            Ok(box client as Box<RtioPipe:Send>)
        }
        n => Err(uv_error_to_io_error(UvError(n)))
    };
    pipe.outgoing.send(msg);
}

impl Drop for PipeListener {
    fn drop(&mut self) {
        let _m = self.fire_homing_missile();
        self.close();
    }
}

// PipeAcceptor implementation and traits

impl RtioUnixAcceptor for PipeAcceptor {
    fn accept(&mut self) -> Result<Box<RtioPipe:Send>, IoError> {
        self.timeout.accept(&self.listener.incoming)
    }

    fn set_timeout(&mut self, timeout_ms: Option<u64>) {
        match timeout_ms {
            None => self.timeout.clear(),
            Some(ms) => self.timeout.set_timeout(ms, &mut *self.listener),
        }
    }
}

impl HomingIO for PipeAcceptor {
    fn home<'r>(&'r mut self) -> &'r mut HomeHandle { &mut self.listener.home }
}

#[cfg(test)]
mod tests {
    use std::rt::rtio::{RtioUnixListener, RtioUnixAcceptor, RtioPipe};
    use std::io::test::next_test_unix;

    use super::{PipeWatcher, PipeListener};
    use super::super::local_loop;

    #[test]
    fn connect_err() {
        match PipeWatcher::connect(local_loop(), &"path/to/nowhere".to_c_str(),
                                   None) {
            Ok(..) => fail!(),
            Err(..) => {}
        }
    }

    #[test]
    fn bind_err() {
        match PipeListener::bind(local_loop(), &"path/to/nowhere".to_c_str()) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.name(), "EACCES".to_owned()),
        }
    }

    #[test]
    fn bind() {
        let p = next_test_unix().to_c_str();
        match PipeListener::bind(local_loop(), &p) {
            Ok(..) => {}
            Err(..) => fail!(),
        }
    }

    #[test] #[should_fail]
    fn bind_fail() {
        let p = next_test_unix().to_c_str();
        let _w = PipeListener::bind(local_loop(), &p).unwrap();
        fail!();
    }

    #[test]
    fn connect() {
        let path = next_test_unix();
        let path2 = path.clone();
        let (tx, rx) = channel();

        spawn(proc() {
            let p = PipeListener::bind(local_loop(), &path2.to_c_str()).unwrap();
            let mut p = p.listen().unwrap();
            tx.send(());
            let mut client = p.accept().unwrap();
            let mut buf = [0];
            assert!(client.read(buf).unwrap() == 1);
            assert_eq!(buf[0], 1);
            assert!(client.write([2]).is_ok());
        });
        rx.recv();
        let mut c = PipeWatcher::connect(local_loop(), &path.to_c_str(), None).unwrap();
        assert!(c.write([1]).is_ok());
        let mut buf = [0];
        assert!(c.read(buf).unwrap() == 1);
        assert_eq!(buf[0], 2);
    }

    #[test] #[should_fail]
    fn connect_fail() {
        let path = next_test_unix();
        let path2 = path.clone();
        let (tx, rx) = channel();

        spawn(proc() {
            let p = PipeListener::bind(local_loop(), &path2.to_c_str()).unwrap();
            let mut p = p.listen().unwrap();
            tx.send(());
            drop(p.accept().unwrap());
        });
        rx.recv();
        let _c = PipeWatcher::connect(local_loop(), &path.to_c_str(), None).unwrap();
        fail!()

    }
}
