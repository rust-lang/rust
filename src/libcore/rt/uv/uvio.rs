// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::*;
use result::*;
use ops::Drop;
use old_iter::CopyableIter;
use cell::{Cell, empty_cell};
use cast::transmute;
use clone::Clone;
use rt::io::IoError;
use rt::io::net::ip::IpAddr;
use rt::uv::*;
use rt::uv::idle::IdleWatcher;
use rt::rtio::*;
use rt::sched::{Scheduler, local_sched};
use rt::io::{standard_error, OtherIoError};
use rt::tube::Tube;

#[cfg(test)] use container::Container;
#[cfg(test)] use uint;
#[cfg(test)] use unstable::run_in_bare_thread;
#[cfg(test)] use rt::test::*;

pub struct UvEventLoop {
    uvio: UvIoFactory
}

pub impl UvEventLoop {
    fn new() -> UvEventLoop {
        UvEventLoop {
            uvio: UvIoFactory(Loop::new())
        }
    }

    /// A convenience constructor
    fn new_scheduler() -> Scheduler {
        Scheduler::new(~UvEventLoop::new())
    }
}

impl Drop for UvEventLoop {
    fn finalize(&self) {
        // XXX: Need mutable finalizer
        let this = unsafe {
            transmute::<&UvEventLoop, &mut UvEventLoop>(self)
        };
        this.uvio.uv_loop().close();
    }
}

impl EventLoop for UvEventLoop {

    fn run(&mut self) {
        self.uvio.uv_loop().run();
    }

    fn callback(&mut self, f: ~fn()) {
        let mut idle_watcher =  IdleWatcher::new(self.uvio.uv_loop());
        do idle_watcher.start |idle_watcher, status| {
            assert!(status.is_none());
            let mut idle_watcher = idle_watcher;
            idle_watcher.stop();
            idle_watcher.close(||());
            f();
        }
    }

    fn callback_ms(&mut self, ms: u64, f: ~fn()) {
        let mut timer =  TimerWatcher::new(self.uvio.uv_loop());
        do timer.start(ms, 0) |timer, status| {
            assert!(status.is_none());
            timer.close(||());
            f();
        }
    }

    fn io<'a>(&'a mut self) -> Option<&'a mut IoFactoryObject> {
        Some(&mut self.uvio)
    }
}

#[test]
fn test_callback_run_once() {
    do run_in_bare_thread {
        let mut event_loop = UvEventLoop::new();
        let mut count = 0;
        let count_ptr: *mut int = &mut count;
        do event_loop.callback {
            unsafe { *count_ptr += 1 }
        }
        event_loop.run();
        assert!(count == 1);
    }
}

pub struct UvIoFactory(Loop);

pub impl UvIoFactory {
    fn uv_loop<'a>(&'a mut self) -> &'a mut Loop {
        match self { &UvIoFactory(ref mut ptr) => ptr }
    }
}

impl IoFactory for UvIoFactory {
    // Connect to an address and return a new stream
    // NB: This blocks the task waiting on the connection.
    // It would probably be better to return a future
    fn tcp_connect(&mut self, addr: IpAddr) -> Result<~RtioTcpStreamObject, IoError> {
        // Create a cell in the task to hold the result. We will fill
        // the cell before resuming the task.
        let result_cell = empty_cell();
        let result_cell_ptr: *Cell<Result<~RtioTcpStreamObject, IoError>> = &result_cell;

        let scheduler = local_sched::take();
        assert!(scheduler.in_task_context());

        // Block this task and take ownership, switch to scheduler context
        do scheduler.deschedule_running_task_and_then |task| {

            rtdebug!("connect: entered scheduler context");
            do local_sched::borrow |scheduler| {
                assert!(!scheduler.in_task_context());
            }
            let mut tcp_watcher = TcpWatcher::new(self.uv_loop());
            let task_cell = Cell(task);

            // Wait for a connection
            do tcp_watcher.connect(addr) |stream_watcher, status| {
                rtdebug!("connect: in connect callback");
                if status.is_none() {
                    rtdebug!("status is none");
                    let res = Ok(~UvTcpStream { watcher: stream_watcher });

                    // Store the stream in the task's stack
                    unsafe { (*result_cell_ptr).put_back(res); }

                    // Context switch
                    let scheduler = local_sched::take();
                    scheduler.resume_task_immediately(task_cell.take());
                } else {
                    rtdebug!("status is some");
                    let task_cell = Cell(task_cell.take());
                    do stream_watcher.close {
                        let res = Err(uv_error_to_io_error(status.get()));
                        unsafe { (*result_cell_ptr).put_back(res); }
                        let scheduler = local_sched::take();
                        scheduler.resume_task_immediately(task_cell.take());
                    }
                };
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn tcp_bind(&mut self, addr: IpAddr) -> Result<~RtioTcpListenerObject, IoError> {
        let mut watcher = TcpWatcher::new(self.uv_loop());
        match watcher.bind(addr) {
            Ok(_) => Ok(~UvTcpListener::new(watcher)),
            Err(uverr) => {
                let scheduler = local_sched::take();
                do scheduler.deschedule_running_task_and_then |task| {
                    let task_cell = Cell(task);
                    do watcher.as_stream().close {
                        let scheduler = local_sched::take();
                        scheduler.resume_task_immediately(task_cell.take());
                    }
                }
                Err(uv_error_to_io_error(uverr))
            }
        }
    }
}

// FIXME #6090: Prefer newtype structs but Drop doesn't work
pub struct UvTcpListener {
    watcher: TcpWatcher,
    listening: bool,
    incoming_streams: Tube<Result<~RtioTcpStreamObject, IoError>>
}

impl UvTcpListener {
    fn new(watcher: TcpWatcher) -> UvTcpListener {
        UvTcpListener {
            watcher: watcher,
            listening: false,
            incoming_streams: Tube::new()
        }
    }

    fn watcher(&self) -> TcpWatcher { self.watcher }
}

impl Drop for UvTcpListener {
    fn finalize(&self) {
        let watcher = self.watcher();
        let scheduler = local_sched::take();
        do scheduler.deschedule_running_task_and_then |task| {
            let task_cell = Cell(task);
            do watcher.as_stream().close {
                let scheduler = local_sched::take();
                scheduler.resume_task_immediately(task_cell.take());
            }
        }
    }
}

impl RtioTcpListener for UvTcpListener {

    fn accept(&mut self) -> Result<~RtioTcpStreamObject, IoError> {
        rtdebug!("entering listen");

        if self.listening {
            return self.incoming_streams.recv();
        }

        self.listening = true;

        let server_tcp_watcher = self.watcher();
        let incoming_streams_cell = Cell(self.incoming_streams.clone());

        let incoming_streams_cell = Cell(incoming_streams_cell.take());
        let mut server_tcp_watcher = server_tcp_watcher;
        do server_tcp_watcher.listen |server_stream_watcher, status| {
            let maybe_stream = if status.is_none() {
                let mut server_stream_watcher = server_stream_watcher;
                let mut loop_ = server_stream_watcher.event_loop();
                let client_tcp_watcher = TcpWatcher::new(&mut loop_);
                let client_tcp_watcher = client_tcp_watcher.as_stream();
                // XXX: Need's to be surfaced in interface
                server_stream_watcher.accept(client_tcp_watcher);
                Ok(~UvTcpStream { watcher: client_tcp_watcher })
            } else {
                Err(standard_error(OtherIoError))
            };

            let mut incoming_streams = incoming_streams_cell.take();
            incoming_streams.send(maybe_stream);
            incoming_streams_cell.put_back(incoming_streams);
        }

        return self.incoming_streams.recv();
    }
}

// FIXME #6090: Prefer newtype structs but Drop doesn't work
pub struct UvTcpStream {
    watcher: StreamWatcher
}

impl UvTcpStream {
    fn watcher(&self) -> StreamWatcher { self.watcher }
}

impl Drop for UvTcpStream {
    fn finalize(&self) {
        rtdebug!("closing tcp stream");
        let watcher = self.watcher();
        let scheduler = local_sched::take();
        do scheduler.deschedule_running_task_and_then |task| {
            let task_cell = Cell(task);
            do watcher.close {
                let scheduler = local_sched::take();
                scheduler.resume_task_immediately(task_cell.take());
            }
        }
    }
}

impl RtioTcpStream for UvTcpStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        let result_cell = empty_cell();
        let result_cell_ptr: *Cell<Result<uint, IoError>> = &result_cell;

        let scheduler = local_sched::take();
        assert!(scheduler.in_task_context());
        let watcher = self.watcher();
        let buf_ptr: *&mut [u8] = &buf;
        do scheduler.deschedule_running_task_and_then |task| {
            rtdebug!("read: entered scheduler context");
            do local_sched::borrow |scheduler| {
                assert!(!scheduler.in_task_context());
            }
            let mut watcher = watcher;
            let task_cell = Cell(task);
            // XXX: We shouldn't reallocate these callbacks every
            // call to read
            let alloc: AllocCallback = |_| unsafe {
                slice_to_uv_buf(*buf_ptr)
            };
            do watcher.read_start(alloc) |watcher, nread, _buf, status| {

                // Stop reading so that no read callbacks are
                // triggered before the user calls `read` again.
                // XXX: Is there a performance impact to calling
                // stop here?
                let mut watcher = watcher;
                watcher.read_stop();

                let result = if status.is_none() {
                    assert!(nread >= 0);
                    Ok(nread as uint)
                } else {
                    Err(uv_error_to_io_error(status.unwrap()))
                };

                unsafe { (*result_cell_ptr).put_back(result); }

                let scheduler = local_sched::take();
                scheduler.resume_task_immediately(task_cell.take());
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let result_cell = empty_cell();
        let result_cell_ptr: *Cell<Result<(), IoError>> = &result_cell;
        let scheduler = local_sched::take();
        assert!(scheduler.in_task_context());
        let watcher = self.watcher();
        let buf_ptr: *&[u8] = &buf;
        do scheduler.deschedule_running_task_and_then |task| {
            let mut watcher = watcher;
            let task_cell = Cell(task);
            let buf = unsafe { slice_to_uv_buf(*buf_ptr) };
            do watcher.write(buf) |_watcher, status| {
                let result = if status.is_none() {
                    Ok(())
                } else {
                    Err(uv_error_to_io_error(status.unwrap()))
                };

                unsafe { (*result_cell_ptr).put_back(result); }

                let scheduler = local_sched::take();
                scheduler.resume_task_immediately(task_cell.take());
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }
}

#[test]
fn test_simple_io_no_connect() {
    do run_in_newsched_task {
        unsafe {
            let io = local_sched::unsafe_borrow_io();
            let addr = next_test_ip4();
            let maybe_chan = (*io).tcp_connect(addr);
            assert!(maybe_chan.is_err());
        }
    }
}

#[test]
fn test_simple_tcp_server_and_client() {
    do run_in_newsched_task {
        let addr = next_test_ip4();

        // Start the server first so it's listening when we connect
        do spawntask_immediately {
            unsafe {
                let io = local_sched::unsafe_borrow_io();
                let mut listener = (*io).tcp_bind(addr).unwrap();
                let mut stream = listener.accept().unwrap();
                let mut buf = [0, .. 2048];
                let nread = stream.read(buf).unwrap();
                assert!(nread == 8);
                for uint::range(0, nread) |i| {
                    rtdebug!("%u", buf[i] as uint);
                    assert!(buf[i] == i as u8);
                }
            }
        }

        do spawntask_immediately {
            unsafe {
                let io = local_sched::unsafe_borrow_io();
                let mut stream = (*io).tcp_connect(addr).unwrap();
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
            }
        }
    }
}

#[test] #[ignore(reason = "busted")]
fn test_read_and_block() {
    do run_in_newsched_task {
        let addr = next_test_ip4();

        do spawntask_immediately {
            let io = unsafe { local_sched::unsafe_borrow_io() };
            let mut listener = unsafe { (*io).tcp_bind(addr).unwrap() };
            let mut stream = listener.accept().unwrap();
            let mut buf = [0, .. 2048];

            let expected = 32;
            let mut current = 0;
            let mut reads = 0;

            while current < expected {
                let nread = stream.read(buf).unwrap();
                for uint::range(0, nread) |i| {
                    let val = buf[i] as uint;
                    assert!(val == current % 8);
                    current += 1;
                }
                reads += 1;

                let scheduler = local_sched::take();
                // Yield to the other task in hopes that it
                // will trigger a read callback while we are
                // not ready for it
                do scheduler.deschedule_running_task_and_then |task| {
                    let task = Cell(task);
                    do local_sched::borrow |scheduler| {
                        scheduler.enqueue_task(task.take());
                    }
                }
            }

            // Make sure we had multiple reads
            assert!(reads > 1);
        }

        do spawntask_immediately {
            unsafe {
                let io = local_sched::unsafe_borrow_io();
                let mut stream = (*io).tcp_connect(addr).unwrap();
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
            }
        }

    }
}

#[test]
fn test_read_read_read() {
    do run_in_newsched_task {
        let addr = next_test_ip4();
        static MAX: uint = 500000;

        do spawntask_immediately {
            unsafe {
                let io = local_sched::unsafe_borrow_io();
                let mut listener = (*io).tcp_bind(addr).unwrap();
                let mut stream = listener.accept().unwrap();
                let buf = [1, .. 2048];
                let mut total_bytes_written = 0;
                while total_bytes_written < MAX {
                    stream.write(buf);
                    total_bytes_written += buf.len();
                }
            }
        }

        do spawntask_immediately {
            unsafe {
                let io = local_sched::unsafe_borrow_io();
                let mut stream = (*io).tcp_connect(addr).unwrap();
                let mut buf = [0, .. 2048];
                let mut total_bytes_read = 0;
                while total_bytes_read < MAX {
                    let nread = stream.read(buf).unwrap();
                    rtdebug!("read %u bytes", nread as uint);
                    total_bytes_read += nread;
                    for uint::range(0, nread) |i| {
                        assert!(buf[i] == 1);
                    }
                }
                rtdebug!("read %u bytes total", total_bytes_read as uint);
            }
        }
    }
}
