// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use c_str::ToCStr;
use cast::transmute;
use cast;
use cell::Cell;
use clone::Clone;
use libc::{c_int, c_uint, c_void};
use ops::Drop;
use option::*;
use ptr;
use result::*;
use rt::io::IoError;
use rt::io::net::ip::{SocketAddr, IpAddr};
use rt::io::{standard_error, OtherIoError};
use rt::local::Local;
use rt::rtio::*;
use rt::sched::Scheduler;
use rt::tube::Tube;
use rt::uv::*;
use rt::uv::idle::IdleWatcher;
use rt::uv::net::{UvIpv4SocketAddr, UvIpv6SocketAddr};
use unstable::sync::Exclusive;

#[cfg(test)] use container::Container;
#[cfg(test)] use unstable::run_in_bare_thread;
#[cfg(test)] use rt::test::{spawntask,
                            next_test_ip4,
                            run_in_newsched_task};
#[cfg(test)] use iterator::{Iterator, range};

enum SocketNameKind {
    TcpPeer,
    Tcp,
    Udp
}

fn socket_name<T, U: Watcher + NativeHandle<*T>>(sk: SocketNameKind,
                                                 handle: U) -> Result<SocketAddr, IoError> {

    let getsockname = match sk {
        TcpPeer => uvll::rust_uv_tcp_getpeername,
        Tcp     => uvll::rust_uv_tcp_getsockname,
        Udp     => uvll::rust_uv_udp_getsockname
    };

    // Allocate a sockaddr_storage
    // since we don't know if it's ipv4 or ipv6
    let r_addr = unsafe { uvll::malloc_sockaddr_storage() };

    let r = unsafe {
        getsockname(handle.native_handle() as *c_void, r_addr as *uvll::sockaddr_storage)
    };

    if r != 0 {
        let status = status_to_maybe_uv_error(handle, r);
        return Err(uv_error_to_io_error(status.unwrap()));
    }

    let addr = unsafe {
        if uvll::is_ip6_addr(r_addr as *uvll::sockaddr) {
            net::uv_socket_addr_to_socket_addr(UvIpv6SocketAddr(r_addr as *uvll::sockaddr_in6))
        } else {
            net::uv_socket_addr_to_socket_addr(UvIpv4SocketAddr(r_addr as *uvll::sockaddr_in))
        }
    };

    unsafe { uvll::free_sockaddr_storage(r_addr); }

    Ok(addr)

}

pub struct UvEventLoop {
    uvio: UvIoFactory
}

impl UvEventLoop {
    pub fn new() -> UvEventLoop {
        UvEventLoop {
            uvio: UvIoFactory(Loop::new())
        }
    }
}

impl Drop for UvEventLoop {
    fn drop(&self) {
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
        do idle_watcher.start |mut idle_watcher, status| {
            assert!(status.is_none());
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

    fn remote_callback(&mut self, f: ~fn()) -> ~RemoteCallbackObject {
        ~UvRemoteCallback::new(self.uvio.uv_loop(), f)
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
        assert_eq!(count, 1);
    }
}

pub struct UvRemoteCallback {
    // The uv async handle for triggering the callback
    async: AsyncWatcher,
    // A flag to tell the callback to exit, set from the dtor. This is
    // almost never contested - only in rare races with the dtor.
    exit_flag: Exclusive<bool>
}

impl UvRemoteCallback {
    pub fn new(loop_: &mut Loop, f: ~fn()) -> UvRemoteCallback {
        let exit_flag = Exclusive::new(false);
        let exit_flag_clone = exit_flag.clone();
        let async = do AsyncWatcher::new(loop_) |watcher, status| {
            assert!(status.is_none());
            f();
            unsafe {
                do exit_flag_clone.with_imm |&should_exit| {
                    if should_exit {
                        watcher.close(||());
                    }
                }
            }
        };
        UvRemoteCallback {
            async: async,
            exit_flag: exit_flag
        }
    }
}

impl RemoteCallback for UvRemoteCallback {
    fn fire(&mut self) { self.async.send() }
}

impl Drop for UvRemoteCallback {
    fn drop(&self) {
        unsafe {
            let this: &mut UvRemoteCallback = cast::transmute_mut(self);
            do this.exit_flag.with |should_exit| {
                // NB: These two things need to happen atomically. Otherwise
                // the event handler could wake up due to a *previous*
                // signal and see the exit flag, destroying the handle
                // before the final send.
                *should_exit = true;
                this.async.send();
            }
        }
    }
}

#[cfg(test)]
mod test_remote {
    use cell::Cell;
    use rt::test::*;
    use rt::thread::Thread;
    use rt::tube::Tube;
    use rt::rtio::EventLoop;
    use rt::local::Local;
    use rt::sched::Scheduler;

    #[test]
    fn test_uv_remote() {
        do run_in_newsched_task {
            let mut tube = Tube::new();
            let tube_clone = tube.clone();
            let remote_cell = Cell::new_empty();
            do Local::borrow::<Scheduler, ()>() |sched| {
                let tube_clone = tube_clone.clone();
                let tube_clone_cell = Cell::new(tube_clone);
                let remote = do sched.event_loop.remote_callback {
                    tube_clone_cell.take().send(1);
                };
                remote_cell.put_back(remote);
            }
            let thread = do Thread::start {
                remote_cell.take().fire();
            };

            assert!(tube.recv() == 1);
            thread.join();
        }
    }
}

pub struct UvIoFactory(Loop);

impl UvIoFactory {
    pub fn uv_loop<'a>(&'a mut self) -> &'a mut Loop {
        match self { &UvIoFactory(ref mut ptr) => ptr }
    }
}

impl IoFactory for UvIoFactory {
    // Connect to an address and return a new stream
    // NB: This blocks the task waiting on the connection.
    // It would probably be better to return a future
    fn tcp_connect(&mut self, addr: SocketAddr) -> Result<~RtioTcpStreamObject, IoError> {
        // Create a cell in the task to hold the result. We will fill
        // the cell before resuming the task.
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<~RtioTcpStreamObject, IoError>> = &result_cell;

        let scheduler = Local::take::<Scheduler>();

        // Block this task and take ownership, switch to scheduler context
        do scheduler.deschedule_running_task_and_then |_, task| {

            rtdebug!("connect: entered scheduler context");
            let mut tcp_watcher = TcpWatcher::new(self.uv_loop());
            let task_cell = Cell::new(task);

            // Wait for a connection
            do tcp_watcher.connect(addr) |stream_watcher, status| {
                rtdebug!("connect: in connect callback");
                if status.is_none() {
                    rtdebug!("status is none");
                    let tcp_watcher =
                        NativeHandle::from_native_handle(stream_watcher.native_handle());
                    let res = Ok(~UvTcpStream(tcp_watcher));

                    // Store the stream in the task's stack
                    unsafe { (*result_cell_ptr).put_back(res); }

                    // Context switch
                    let scheduler = Local::take::<Scheduler>();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                } else {
                    rtdebug!("status is some");
                    let task_cell = Cell::new(task_cell.take());
                    do stream_watcher.close {
                        let res = Err(uv_error_to_io_error(status.unwrap()));
                        unsafe { (*result_cell_ptr).put_back(res); }
                        let scheduler = Local::take::<Scheduler>();
                        scheduler.resume_blocked_task_immediately(task_cell.take());
                    }
                };
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn tcp_bind(&mut self, addr: SocketAddr) -> Result<~RtioTcpListenerObject, IoError> {
        let mut watcher = TcpWatcher::new(self.uv_loop());
        match watcher.bind(addr) {
            Ok(_) => Ok(~UvTcpListener::new(watcher)),
            Err(uverr) => {
                let scheduler = Local::take::<Scheduler>();
                do scheduler.deschedule_running_task_and_then |_, task| {
                    let task_cell = Cell::new(task);
                    do watcher.as_stream().close {
                        let scheduler = Local::take::<Scheduler>();
                        scheduler.resume_blocked_task_immediately(task_cell.take());
                    }
                }
                Err(uv_error_to_io_error(uverr))
            }
        }
    }

    fn udp_bind(&mut self, addr: SocketAddr) -> Result<~RtioUdpSocketObject, IoError> {
        let mut watcher = UdpWatcher::new(self.uv_loop());
        match watcher.bind(addr) {
            Ok(_) => Ok(~UvUdpSocket(watcher)),
            Err(uverr) => {
                let scheduler = Local::take::<Scheduler>();
                do scheduler.deschedule_running_task_and_then |_, task| {
                    let task_cell = Cell::new(task);
                    do watcher.close {
                        let scheduler = Local::take::<Scheduler>();
                        scheduler.resume_blocked_task_immediately(task_cell.take());
                    }
                }
                Err(uv_error_to_io_error(uverr))
            }
        }
    }

    fn timer_init(&mut self) -> Result<~RtioTimerObject, IoError> {
        Ok(~UvTimer(TimerWatcher::new(self.uv_loop())))
    }
}

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
    fn drop(&self) {
        let watcher = self.watcher();
        let scheduler = Local::take::<Scheduler>();
        do scheduler.deschedule_running_task_and_then |_, task| {
            let task_cell = Cell::new(task);
            do watcher.as_stream().close {
                let scheduler = Local::take::<Scheduler>();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }
    }
}

impl RtioSocket for UvTcpListener {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        socket_name(Tcp, self.watcher)
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
        let incoming_streams_cell = Cell::new(self.incoming_streams.clone());

        let incoming_streams_cell = Cell::new(incoming_streams_cell.take());
        let mut server_tcp_watcher = server_tcp_watcher;
        do server_tcp_watcher.listen |mut server_stream_watcher, status| {
            let maybe_stream = if status.is_none() {
                let mut loop_ = server_stream_watcher.event_loop();
                let client_tcp_watcher = TcpWatcher::new(&mut loop_);
                // XXX: Need's to be surfaced in interface
                server_stream_watcher.accept(client_tcp_watcher.as_stream());
                Ok(~UvTcpStream(client_tcp_watcher))
            } else {
                Err(standard_error(OtherIoError))
            };

            let mut incoming_streams = incoming_streams_cell.take();
            incoming_streams.send(maybe_stream);
            incoming_streams_cell.put_back(incoming_streams);
        }

        return self.incoming_streams.recv();
    }

    fn accept_simultaneously(&mut self) -> Result<(), IoError> {
        let r = unsafe {
            uvll::rust_uv_tcp_simultaneous_accepts(self.watcher.native_handle(), 1 as c_int)
        };

        match status_to_maybe_uv_error(self.watcher, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn dont_accept_simultaneously(&mut self) -> Result<(), IoError> {
        let r = unsafe {
            uvll::rust_uv_tcp_simultaneous_accepts(self.watcher.native_handle(), 0 as c_int)
        };

        match status_to_maybe_uv_error(self.watcher, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }
}

pub struct UvTcpStream(TcpWatcher);

impl Drop for UvTcpStream {
    fn drop(&self) {
        rtdebug!("closing tcp stream");
        let scheduler = Local::take::<Scheduler>();
        do scheduler.deschedule_running_task_and_then |_, task| {
            let task_cell = Cell::new(task);
            do self.as_stream().close {
                let scheduler = Local::take::<Scheduler>();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }
    }
}

impl RtioSocket for UvTcpStream {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        socket_name(Tcp, **self)
    }
}

impl RtioTcpStream for UvTcpStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<uint, IoError>> = &result_cell;

        let scheduler = Local::take::<Scheduler>();
        let buf_ptr: *&mut [u8] = &buf;
        do scheduler.deschedule_running_task_and_then |_sched, task| {
            rtdebug!("read: entered scheduler context");
            let task_cell = Cell::new(task);
            // XXX: We shouldn't reallocate these callbacks every
            // call to read
            let alloc: AllocCallback = |_| unsafe {
                slice_to_uv_buf(*buf_ptr)
            };
            let mut watcher = self.as_stream();
            do watcher.read_start(alloc) |mut watcher, nread, _buf, status| {

                // Stop reading so that no read callbacks are
                // triggered before the user calls `read` again.
                // XXX: Is there a performance impact to calling
                // stop here?
                watcher.read_stop();

                let result = if status.is_none() {
                    assert!(nread >= 0);
                    Ok(nread as uint)
                } else {
                    Err(uv_error_to_io_error(status.unwrap()))
                };

                unsafe { (*result_cell_ptr).put_back(result); }

                let scheduler = Local::take::<Scheduler>();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<(), IoError>> = &result_cell;
        let scheduler = Local::take::<Scheduler>();
        let buf_ptr: *&[u8] = &buf;
        do scheduler.deschedule_running_task_and_then |_, task| {
            let task_cell = Cell::new(task);
            let buf = unsafe { slice_to_uv_buf(*buf_ptr) };
            let mut watcher = self.as_stream();
            do watcher.write(buf) |_watcher, status| {
                let result = if status.is_none() {
                    Ok(())
                } else {
                    Err(uv_error_to_io_error(status.unwrap()))
                };

                unsafe { (*result_cell_ptr).put_back(result); }

                let scheduler = Local::take::<Scheduler>();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn peer_name(&mut self) -> Result<SocketAddr, IoError> {
        socket_name(TcpPeer, **self)
    }

    fn control_congestion(&mut self) -> Result<(), IoError> {
        let r = unsafe {
            uvll::rust_uv_tcp_nodelay(self.native_handle(), 0 as c_int)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn nodelay(&mut self) -> Result<(), IoError> {
        let r = unsafe {
            uvll::rust_uv_tcp_nodelay(self.native_handle(), 1 as c_int)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn keepalive(&mut self, delay_in_seconds: uint) -> Result<(), IoError> {
        let r = unsafe {
            uvll::rust_uv_tcp_keepalive(self.native_handle(), 1 as c_int,
                                        delay_in_seconds as c_uint)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn letdie(&mut self) -> Result<(), IoError> {
        let r = unsafe {
            uvll::rust_uv_tcp_keepalive(self.native_handle(), 0 as c_int, 0 as c_uint)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }
}

pub struct UvUdpSocket(UdpWatcher);

impl Drop for UvUdpSocket {
    fn drop(&self) {
        rtdebug!("closing udp socket");
        let scheduler = Local::take::<Scheduler>();
        do scheduler.deschedule_running_task_and_then |_, task| {
            let task_cell = Cell::new(task);
            do self.close {
                let scheduler = Local::take::<Scheduler>();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }
    }
}

impl RtioSocket for UvUdpSocket {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        socket_name(Udp, **self)
    }
}

impl RtioUdpSocket for UvUdpSocket {
    fn recvfrom(&mut self, buf: &mut [u8]) -> Result<(uint, SocketAddr), IoError> {
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<(uint, SocketAddr), IoError>> = &result_cell;

        let scheduler = Local::take::<Scheduler>();
        let buf_ptr: *&mut [u8] = &buf;
        do scheduler.deschedule_running_task_and_then |_sched, task| {
            rtdebug!("recvfrom: entered scheduler context");
            let task_cell = Cell::new(task);
            let alloc: AllocCallback = |_| unsafe { slice_to_uv_buf(*buf_ptr) };
            do self.recv_start(alloc) |mut watcher, nread, _buf, addr, flags, status| {
                let _ = flags; // XXX add handling for partials?

                watcher.recv_stop();

                let result = match status {
                    None => {
                        assert!(nread >= 0);
                        Ok((nread as uint, addr))
                    }
                    Some(err) => Err(uv_error_to_io_error(err))
                };

                unsafe { (*result_cell_ptr).put_back(result); }

                let scheduler = Local::take::<Scheduler>();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn sendto(&mut self, buf: &[u8], dst: SocketAddr) -> Result<(), IoError> {
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<(), IoError>> = &result_cell;
        let scheduler = Local::take::<Scheduler>();
        let buf_ptr: *&[u8] = &buf;
        do scheduler.deschedule_running_task_and_then |_, task| {
            let task_cell = Cell::new(task);
            let buf = unsafe { slice_to_uv_buf(*buf_ptr) };
            do self.send(buf, dst) |_watcher, status| {

                let result = match status {
                    None => Ok(()),
                    Some(err) => Err(uv_error_to_io_error(err)),
                };

                unsafe { (*result_cell_ptr).put_back(result); }

                let scheduler = Local::take::<Scheduler>();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn join_multicast(&mut self, multi: IpAddr) -> Result<(), IoError> {
        let r = unsafe {
            do multi.to_str().to_c_str().with_ref |m_addr| {
                uvll::udp_set_membership(self.native_handle(), m_addr,
                                         ptr::null(), uvll::UV_JOIN_GROUP)
            }
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn leave_multicast(&mut self, multi: IpAddr) -> Result<(), IoError> {
        let r = unsafe {
            do multi.to_str().to_c_str().with_ref |m_addr| {
                uvll::udp_set_membership(self.native_handle(), m_addr,
                                         ptr::null(), uvll::UV_LEAVE_GROUP)
            }
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn loop_multicast_locally(&mut self) -> Result<(), IoError> {
        let r = unsafe {
            uvll::udp_set_multicast_loop(self.native_handle(), 1 as c_int)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn dont_loop_multicast_locally(&mut self) -> Result<(), IoError> {
        let r = unsafe {
            uvll::udp_set_multicast_loop(self.native_handle(), 0 as c_int)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn multicast_time_to_live(&mut self, ttl: int) -> Result<(), IoError> {
        let r = unsafe {
            uvll::udp_set_multicast_ttl(self.native_handle(), ttl as c_int)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn time_to_live(&mut self, ttl: int) -> Result<(), IoError> {
        let r = unsafe {
            uvll::udp_set_ttl(self.native_handle(), ttl as c_int)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn hear_broadcasts(&mut self) -> Result<(), IoError> {
        let r = unsafe {
            uvll::udp_set_broadcast(self.native_handle(), 1 as c_int)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }

    fn ignore_broadcasts(&mut self) -> Result<(), IoError> {
        let r = unsafe {
            uvll::udp_set_broadcast(self.native_handle(), 0 as c_int)
        };

        match status_to_maybe_uv_error(**self, r) {
            Some(err) => Err(uv_error_to_io_error(err)),
            None => Ok(())
        }
    }
}

pub struct UvTimer(timer::TimerWatcher);

impl UvTimer {
    fn new(w: timer::TimerWatcher) -> UvTimer {
        UvTimer(w)
    }
}

impl Drop for UvTimer {
    fn drop(&self) {
        rtdebug!("closing UvTimer");
        let scheduler = Local::take::<Scheduler>();
        do scheduler.deschedule_running_task_and_then |_, task| {
            let task_cell = Cell::new(task);
            do self.close {
                let scheduler = Local::take::<Scheduler>();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }
    }
}

impl RtioTimer for UvTimer {
    fn sleep(&self, msecs: u64) {
        let scheduler = Local::take::<Scheduler>();
        do scheduler.deschedule_running_task_and_then |_sched, task| {
            rtdebug!("sleep: entered scheduler context");
            let task_cell = Cell::new(task);
            let mut watcher = **self;
            do watcher.start(msecs, 0) |_, status| {
                assert!(status.is_none());
                let scheduler = Local::take::<Scheduler>();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }
        let mut w = **self;
        w.stop();
    }
}

#[test]
fn test_simple_io_no_connect() {
    do run_in_newsched_task {
        unsafe {
            let io = Local::unsafe_borrow::<IoFactoryObject>();
            let addr = next_test_ip4();
            let maybe_chan = (*io).tcp_connect(addr);
            assert!(maybe_chan.is_err());
        }
    }
}

#[test]
fn test_simple_udp_io_bind_only() {
    do run_in_newsched_task {
        unsafe {
            let io = Local::unsafe_borrow::<IoFactoryObject>();
            let addr = next_test_ip4();
            let maybe_socket = (*io).udp_bind(addr);
            assert!(maybe_socket.is_ok());
        }
    }
}

#[test]
fn test_simple_tcp_server_and_client() {
    do run_in_newsched_task {
        let addr = next_test_ip4();

        // Start the server first so it's listening when we connect
        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
                let mut listener = (*io).tcp_bind(addr).unwrap();
                let mut stream = listener.accept().unwrap();
                let mut buf = [0, .. 2048];
                let nread = stream.read(buf).unwrap();
                assert_eq!(nread, 8);
                for i in range(0u, nread) {
                    rtdebug!("%u", buf[i] as uint);
                    assert_eq!(buf[i], i as u8);
                }
            }
        }

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
                let mut stream = (*io).tcp_connect(addr).unwrap();
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
            }
        }
    }
}

#[test]
fn test_simple_udp_server_and_client() {
    do run_in_newsched_task {
        let server_addr = next_test_ip4();
        let client_addr = next_test_ip4();

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
                let mut server_socket = (*io).udp_bind(server_addr).unwrap();
                let mut buf = [0, .. 2048];
                let (nread,src) = server_socket.recvfrom(buf).unwrap();
                assert_eq!(nread, 8);
                for i in range(0u, nread) {
                    rtdebug!("%u", buf[i] as uint);
                    assert_eq!(buf[i], i as u8);
                }
                assert_eq!(src, client_addr);
            }
        }

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
                let mut client_socket = (*io).udp_bind(client_addr).unwrap();
                client_socket.sendto([0, 1, 2, 3, 4, 5, 6, 7], server_addr);
            }
        }
    }
}

#[test] #[ignore(reason = "busted")]
fn test_read_and_block() {
    do run_in_newsched_task {
        let addr = next_test_ip4();

        do spawntask {
            let io = unsafe { Local::unsafe_borrow::<IoFactoryObject>() };
            let mut listener = unsafe { (*io).tcp_bind(addr).unwrap() };
            let mut stream = listener.accept().unwrap();
            let mut buf = [0, .. 2048];

            let expected = 32;
            let mut current = 0;
            let mut reads = 0;

            while current < expected {
                let nread = stream.read(buf).unwrap();
                for i in range(0u, nread) {
                    let val = buf[i] as uint;
                    assert_eq!(val, current % 8);
                    current += 1;
                }
                reads += 1;

                let scheduler = Local::take::<Scheduler>();
                // Yield to the other task in hopes that it
                // will trigger a read callback while we are
                // not ready for it
                do scheduler.deschedule_running_task_and_then |sched, task| {
                    let task = Cell::new(task);
                    sched.enqueue_blocked_task(task.take());
                }
            }

            // Make sure we had multiple reads
            assert!(reads > 1);
        }

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
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

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
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

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
                let mut stream = (*io).tcp_connect(addr).unwrap();
                let mut buf = [0, .. 2048];
                let mut total_bytes_read = 0;
                while total_bytes_read < MAX {
                    let nread = stream.read(buf).unwrap();
                    rtdebug!("read %u bytes", nread as uint);
                    total_bytes_read += nread;
                    for i in range(0u, nread) {
                        assert_eq!(buf[i], 1);
                    }
                }
                rtdebug!("read %u bytes total", total_bytes_read as uint);
            }
        }
    }
}

#[test]
fn test_udp_twice() {
    do run_in_newsched_task {
        let server_addr = next_test_ip4();
        let client_addr = next_test_ip4();

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
                let mut client = (*io).udp_bind(client_addr).unwrap();
                assert!(client.sendto([1], server_addr).is_ok());
                assert!(client.sendto([2], server_addr).is_ok());
            }
        }

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
                let mut server = (*io).udp_bind(server_addr).unwrap();
                let mut buf1 = [0];
                let mut buf2 = [0];
                let (nread1, src1) = server.recvfrom(buf1).unwrap();
                let (nread2, src2) = server.recvfrom(buf2).unwrap();
                assert_eq!(nread1, 1);
                assert_eq!(nread2, 1);
                assert_eq!(src1, client_addr);
                assert_eq!(src2, client_addr);
                assert_eq!(buf1[0], 1);
                assert_eq!(buf2[0], 2);
            }
        }
    }
}

#[test]
fn test_udp_many_read() {
    do run_in_newsched_task {
        let server_out_addr = next_test_ip4();
        let server_in_addr = next_test_ip4();
        let client_out_addr = next_test_ip4();
        let client_in_addr = next_test_ip4();
        static MAX: uint = 500_000;

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
                let mut server_out = (*io).udp_bind(server_out_addr).unwrap();
                let mut server_in = (*io).udp_bind(server_in_addr).unwrap();
                let msg = [1, .. 2048];
                let mut total_bytes_sent = 0;
                let mut buf = [1];
                while buf[0] == 1 {
                    // send more data
                    assert!(server_out.sendto(msg, client_in_addr).is_ok());
                    total_bytes_sent += msg.len();
                    // check if the client has received enough
                    let res = server_in.recvfrom(buf);
                    assert!(res.is_ok());
                    let (nread, src) = res.unwrap();
                    assert_eq!(nread, 1);
                    assert_eq!(src, client_out_addr);
                }
                assert!(total_bytes_sent >= MAX);
            }
        }

        do spawntask {
            unsafe {
                let io = Local::unsafe_borrow::<IoFactoryObject>();
                let mut client_out = (*io).udp_bind(client_out_addr).unwrap();
                let mut client_in = (*io).udp_bind(client_in_addr).unwrap();
                let mut total_bytes_recv = 0;
                let mut buf = [0, .. 2048];
                while total_bytes_recv < MAX {
                    // ask for more
                    assert!(client_out.sendto([1], server_in_addr).is_ok());
                    // wait for data
                    let res = client_in.recvfrom(buf);
                    assert!(res.is_ok());
                    let (nread, src) = res.unwrap();
                    assert_eq!(src, server_out_addr);
                    total_bytes_recv += nread;
                    for i in range(0u, nread) {
                        assert_eq!(buf[i], 1);
                    }
                }
                // tell the server we're done
                assert!(client_out.sendto([0], server_in_addr).is_ok());
            }
        }
    }
}

fn test_timer_sleep_simple_impl() {
    unsafe {
        let io = Local::unsafe_borrow::<IoFactoryObject>();
        let timer = (*io).timer_init();
        match timer {
            Ok(t) => t.sleep(1),
            Err(_) => assert!(false)
        }
    }
}
#[test]
fn test_timer_sleep_simple() {
    do run_in_newsched_task {
        test_timer_sleep_simple_impl();
    }
}
