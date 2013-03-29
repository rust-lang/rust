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

use super::uv::*;
use super::rtio::*;
use ops::Drop;
use cell::{Cell, empty_cell};
use cast::transmute;
use super::sched::Scheduler;

#[cfg(test)] use super::sched::Task;
#[cfg(test)] use unstable::run_in_bare_thread;
#[cfg(test)] use uint;

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
        let self = unsafe {
            transmute::<&UvEventLoop, &mut UvEventLoop>(self)
        };
        let mut uv_loop = self.uvio.uv_loop();
        uv_loop.close();
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
            idle_watcher.close();
            f();
        }
    }

    fn io(&mut self) -> Option<&'self mut IoFactoryObject> {
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
    fn uv_loop(&mut self) -> &'self mut Loop {
        match self { &UvIoFactory(ref mut ptr) => ptr }
    }
}

impl IoFactory for UvIoFactory {
    // Connect to an address and return a new stream
    // NB: This blocks the task waiting on the connection.
    // It would probably be better to return a future
    fn connect(&mut self, addr: IpAddr) -> Option<~StreamObject> {
        // Create a cell in the task to hold the result. We will fill
        // the cell before resuming the task.
        let result_cell = empty_cell();
        let result_cell_ptr: *Cell<Option<~StreamObject>> = &result_cell;

        do Scheduler::local |scheduler| {
            assert!(scheduler.in_task_context());

            // Block this task and take ownership, switch to scheduler context
            do scheduler.block_running_task_and_then |scheduler, task| {

                rtdebug!("connect: entered scheduler context");
                assert!(!scheduler.in_task_context());
                let mut tcp_watcher = TcpWatcher::new(self.uv_loop());
                let task_cell = Cell(task);

                // Wait for a connection
                do tcp_watcher.connect(addr) |stream_watcher, status| {
                    rtdebug!("connect: in connect callback");
                    let maybe_stream = if status.is_none() {
                        rtdebug!("status is none");
                        Some(~UvStream(stream_watcher))
                    } else {
                        rtdebug!("status is some");
                        stream_watcher.close(||());
                        None
                    };

                    // Store the stream in the task's stack
                    unsafe { (*result_cell_ptr).put_back(maybe_stream); }

                    // Context switch
                    do Scheduler::local |scheduler| {
                        scheduler.resume_task_immediately(task_cell.take());
                    }
                }
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn bind(&mut self, addr: IpAddr) -> Option<~TcpListenerObject> {
        let mut watcher = TcpWatcher::new(self.uv_loop());
        watcher.bind(addr);
        return Some(~UvTcpListener(watcher));
    }
}

pub struct UvTcpListener(TcpWatcher);

impl UvTcpListener {
    fn watcher(&self) -> TcpWatcher {
        match self { &UvTcpListener(w) => w }
    }

    fn close(&self) {
        // XXX: Need to wait until close finishes before returning
        self.watcher().as_stream().close(||());
    }
}

impl Drop for UvTcpListener {
    fn finalize(&self) {
        // XXX: Again, this never gets called. Use .close() instead
        //self.watcher().as_stream().close(||());
    }
}

impl TcpListener for UvTcpListener {

    fn listen(&mut self) -> Option<~StreamObject> {
        rtdebug!("entering listen");
        let result_cell = empty_cell();
        let result_cell_ptr: *Cell<Option<~StreamObject>> = &result_cell;

        let server_tcp_watcher = self.watcher();

        do Scheduler::local |scheduler| {
            assert!(scheduler.in_task_context());

            do scheduler.block_running_task_and_then |_, task| {
                let task_cell = Cell(task);
                let mut server_tcp_watcher = server_tcp_watcher;
                do server_tcp_watcher.listen |server_stream_watcher, status| {
                    let maybe_stream = if status.is_none() {
                        let mut server_stream_watcher = server_stream_watcher;
                        let mut loop_ = loop_from_watcher(&server_stream_watcher);
                        let mut client_tcp_watcher = TcpWatcher::new(&mut loop_);
                        let mut client_tcp_watcher = client_tcp_watcher.as_stream();
                        // XXX: Need's to be surfaced in interface
                        server_stream_watcher.accept(client_tcp_watcher);
                        Some(~UvStream::new(client_tcp_watcher))
                    } else {
                        None
                    };

                    unsafe { (*result_cell_ptr).put_back(maybe_stream); }

                    rtdebug!("resuming task from listen");
                    // Context switch
                    do Scheduler::local |scheduler| {
                        scheduler.resume_task_immediately(task_cell.take());
                    }
                }
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }
}

pub struct UvStream(StreamWatcher);

impl UvStream {
    fn new(watcher: StreamWatcher) -> UvStream {
        UvStream(watcher)
    }

    fn watcher(&self) -> StreamWatcher {
        match self { &UvStream(w) => w }
    }

    // XXX: finalize isn't working for ~UvStream???
    fn close(&self) {
        // XXX: Need to wait until this finishes before returning
        self.watcher().close(||());
    }
}

impl Drop for UvStream {
    fn finalize(&self) {
        rtdebug!("closing stream");
        //self.watcher().close(||());
    }
}

impl Stream for UvStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, ()> {
        let result_cell = empty_cell();
        let result_cell_ptr: *Cell<Result<uint, ()>> = &result_cell;

        do Scheduler::local |scheduler| {
            assert!(scheduler.in_task_context());
            let watcher = self.watcher();
            let buf_ptr: *&mut [u8] = &buf;
            do scheduler.block_running_task_and_then |scheduler, task| {
                rtdebug!("read: entered scheduler context");
                assert!(!scheduler.in_task_context());
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
                        Err(())
                    };

                    unsafe { (*result_cell_ptr).put_back(result); }

                    do Scheduler::local |scheduler| {
                        scheduler.resume_task_immediately(task_cell.take());
                    }
                }
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), ()> {
        let result_cell = empty_cell();
        let result_cell_ptr: *Cell<Result<(), ()>> = &result_cell;
        do Scheduler::local |scheduler| {
            assert!(scheduler.in_task_context());
            let watcher = self.watcher();
            let buf_ptr: *&[u8] = &buf;
            do scheduler.block_running_task_and_then |_, task| {
                let mut watcher = watcher;
                let task_cell = Cell(task);
                let buf = unsafe { &*buf_ptr };
                // XXX: OMGCOPIES
                let buf = buf.to_vec();
                do watcher.write(buf) |_watcher, status| {
                    let result = if status.is_none() {
                        Ok(())
                    } else {
                        Err(())
                    };

                    unsafe { (*result_cell_ptr).put_back(result); }

                    do Scheduler::local |scheduler| {
                        scheduler.resume_task_immediately(task_cell.take());
                    }
                }
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }
}

#[test]
#[ignore(reason = "ffi struct issues")]
fn test_simple_io_no_connect() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let task = ~do Task::new(&mut sched.stack_pool) {
            do Scheduler::local |sched| {
                let io = sched.event_loop.io().unwrap();
                let addr = Ipv4(127, 0, 0, 1, 2926);
                let maybe_chan = io.connect(addr);
                assert!(maybe_chan.is_none());
            }
        };
        sched.task_queue.push_back(task);
        sched.run();
    }
}

#[test]
#[ignore(reason = "ffi struct issues")]
fn test_simple_tcp_server_and_client() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let addr = Ipv4(127, 0, 0, 1, 2929);

        let client_task = ~do Task::new(&mut sched.stack_pool) {
            do Scheduler::local |sched| {
                let io = sched.event_loop.io().unwrap();
                let mut stream = io.connect(addr).unwrap();
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.close();
            }
        };

        let server_task = ~do Task::new(&mut sched.stack_pool) {
            do Scheduler::local |sched| {
                let io = sched.event_loop.io().unwrap();
                let mut listener = io.bind(addr).unwrap();
                let mut stream = listener.listen().unwrap();
                let mut buf = [0, .. 2048];
                let nread = stream.read(buf).unwrap();
                assert!(nread == 8);
                for uint::range(0, nread) |i| {
                    rtdebug!("%u", buf[i] as uint);
                    assert!(buf[i] == i as u8);
                }
                stream.close();
                listener.close();
            }
        };

        // Start the server first so it listens before the client connects
        sched.task_queue.push_back(server_task);
        sched.task_queue.push_back(client_task);
        sched.run();
    }
}

#[test] #[ignore(reason = "busted")]
fn test_read_and_block() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let addr = Ipv4(127, 0, 0, 1, 2930);

        let client_task = ~do Task::new(&mut sched.stack_pool) {
            do Scheduler::local |sched| {
                let io = sched.event_loop.io().unwrap();
                let mut stream = io.connect(addr).unwrap();
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.close();
            }
        };

        let server_task = ~do Task::new(&mut sched.stack_pool) {
            do Scheduler::local |sched| {
                let io = sched.event_loop.io().unwrap();
                let mut listener = io.bind(addr).unwrap();
                let mut stream = listener.listen().unwrap();
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

                    do Scheduler::local |scheduler| {
                        // Yield to the other task in hopes that it
                        // will trigger a read callback while we are
                        // not ready for it
                        do scheduler.block_running_task_and_then |scheduler, task| {
                            scheduler.task_queue.push_back(task);
                        }
                    }
                }

                // Make sure we had multiple reads
                assert!(reads > 1);

                stream.close();
                listener.close();
            }
        };

        // Start the server first so it listens before the client connects
        sched.task_queue.push_back(server_task);
        sched.task_queue.push_back(client_task);
        sched.run();
    }
}

#[test] #[ignore(reason = "needs server")]
fn test_read_read_read() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let addr = Ipv4(127, 0, 0, 1, 2931);

        let client_task = ~do Task::new(&mut sched.stack_pool) {
            do Scheduler::local |sched| {
                let io = sched.event_loop.io().unwrap();
                let mut stream = io.connect(addr).unwrap();
                let mut buf = [0, .. 2048];
                let mut total_bytes_read = 0;
                while total_bytes_read < 500000000 {
                    let nread = stream.read(buf).unwrap();
                    rtdebug!("read %u bytes", nread as uint);
                    total_bytes_read += nread;
                }
                rtdebug_!("read %u bytes total", total_bytes_read as uint);
                stream.close();
            }
        };

        sched.task_queue.push_back(client_task);
        sched.run();
    }
}
