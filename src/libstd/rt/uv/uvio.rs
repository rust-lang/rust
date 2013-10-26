// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use c_str::{ToCStr, CString};
use cast::transmute;
use cast;
use cell::Cell;
use clone::Clone;
use comm::{SendDeferred, SharedChan};
use libc::{c_int, c_uint, c_void, pid_t};
use ops::Drop;
use option::*;
use ptr;
use str;
use result::*;
use rt::io::IoError;
use rt::io::net::ip::{SocketAddr, IpAddr};
use rt::io::{standard_error, OtherIoError, SeekStyle, SeekSet, SeekCur, SeekEnd};
use rt::io::process::ProcessConfig;
use rt::kill::BlockedTask;
use rt::local::Local;
use rt::rtio::*;
use rt::sched::{Scheduler, SchedHandle};
use rt::tube::Tube;
use rt::task::Task;
use rt::uv::*;
use rt::uv::idle::IdleWatcher;
use rt::uv::net::{UvIpv4SocketAddr, UvIpv6SocketAddr};
use rt::uv::addrinfo::{GetAddrInfoRequest, accum_addrinfo};
use unstable::sync::Exclusive;
use path::{GenericPath, Path};
use libc::{lseek, off_t, O_CREAT, O_APPEND, O_TRUNC, O_RDWR, O_RDONLY, O_WRONLY,
          S_IRUSR, S_IWUSR, S_IRWXU};
use rt::io::{FileMode, FileAccess, OpenOrCreate, Open, Create,
             CreateOrTruncate, Append, Truncate, Read, Write, ReadWrite,
             FileStat};
use rt::io::signal::Signum;
use task;
use ai = rt::io::net::addrinfo;

#[cfg(test)] use container::Container;
#[cfg(test)] use unstable::run_in_bare_thread;
#[cfg(test)] use rt::test::{spawntask,
                            next_test_ip4,
                            run_in_mt_newsched_task};
#[cfg(test)] use iter::{Iterator, range};
#[cfg(test)] use rt::comm::oneshot;

// XXX we should not be calling uvll functions in here.

trait HomingIO {

    fn home<'r>(&'r mut self) -> &'r mut SchedHandle;

    /// This function will move tasks to run on their home I/O scheduler. Note
    /// that this function does *not* pin the task to the I/O scheduler, but
    /// rather it simply moves it to running on the I/O scheduler.
    fn go_to_IO_home(&mut self) -> uint {
        use rt::sched::RunOnce;

        let current_sched_id = do Local::borrow |sched: &mut Scheduler| {
            sched.sched_id()
        };

        // Only need to invoke a context switch if we're not on the right
        // scheduler.
        if current_sched_id != self.home().sched_id {
            do task::unkillable { // FIXME(#8674)
                let scheduler: ~Scheduler = Local::take();
                do scheduler.deschedule_running_task_and_then |_, task| {
                    /* FIXME(#8674) if the task was already killed then wake
                     * will return None. In that case, the home pointer will
                     * never be set.
                     *
                     * RESOLUTION IDEA: Since the task is dead, we should
                     * just abort the IO action.
                     */
                    do task.wake().map |task| {
                        self.home().send(RunOnce(task));
                    };
                }
            }
        }

        self.home().sched_id
    }

    // XXX: dummy self parameter
    fn restore_original_home(_: Option<Self>, io_home: uint) {
        // It would truly be a sad day if we had moved off the home I/O
        // scheduler while we were doing I/O.
        assert_eq!(Local::borrow(|sched: &mut Scheduler| sched.sched_id()),
                   io_home);

        // If we were a homed task, then we must send ourselves back to the
        // original scheduler. Otherwise, we can just return and keep running
        if !Task::on_appropriate_sched() {
            do task::unkillable { // FIXME(#8674)
                let scheduler: ~Scheduler = Local::take();
                do scheduler.deschedule_running_task_and_then |_, task| {
                    do task.wake().map |task| {
                        Scheduler::run_task(task);
                    };
                }
            }
        }
    }

    fn home_for_io<A>(&mut self, io: &fn(&mut Self) -> A) -> A {
        let home = self.go_to_IO_home();
        let a = io(self); // do IO
        HomingIO::restore_original_home(None::<Self>, home);
        a // return the result of the IO
    }

    fn home_for_io_consume<A>(self, io: &fn(Self) -> A) -> A {
        let mut this = self;
        let home = this.go_to_IO_home();
        let a = io(this); // do IO
        HomingIO::restore_original_home(None::<Self>, home);
        a // return the result of the IO
    }

    fn home_for_io_with_sched<A>(&mut self, io_sched: &fn(&mut Self, ~Scheduler) -> A) -> A {
        let home = self.go_to_IO_home();
        let a = do task::unkillable { // FIXME(#8674)
            let scheduler: ~Scheduler = Local::take();
            io_sched(self, scheduler) // do IO and scheduling action
        };
        HomingIO::restore_original_home(None::<Self>, home);
        a // return result of IO
    }
}

// get a handle for the current scheduler
macro_rules! get_handle_to_current_scheduler(
    () => (do Local::borrow |sched: &mut Scheduler| { sched.make_handle() })
)

enum SocketNameKind {
    TcpPeer,
    Tcp,
    Udp
}

fn socket_name<T, U: Watcher + NativeHandle<*T>>(sk: SocketNameKind,
                                                 handle: U) -> Result<SocketAddr, IoError> {
    let getsockname = match sk {
        TcpPeer => uvll::tcp_getpeername,
        Tcp     => uvll::tcp_getsockname,
        Udp     => uvll::udp_getsockname,
    };

    // Allocate a sockaddr_storage
    // since we don't know if it's ipv4 or ipv6
    let r_addr = unsafe { uvll::malloc_sockaddr_storage() };

    let r = unsafe {
        getsockname(handle.native_handle() as *c_void, r_addr as *uvll::sockaddr_storage)
    };

    if r != 0 {
        let status = status_to_maybe_uv_error(r);
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

// Obviously an Event Loop is always home.
pub struct UvEventLoop {
    priv uvio: UvIoFactory
}

impl UvEventLoop {
    pub fn new() -> UvEventLoop {
        UvEventLoop {
            uvio: UvIoFactory(Loop::new())
        }
    }
}

impl Drop for UvEventLoop {
    fn drop(&mut self) {
        self.uvio.uv_loop().close();
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

    fn pausible_idle_callback(&mut self) -> ~PausibleIdleCallback {
        let idle_watcher = IdleWatcher::new(self.uvio.uv_loop());
        ~UvPausibleIdleCallback {
            watcher: idle_watcher,
            idle_flag: false,
            closed: false
        } as ~PausibleIdleCallback
    }

    fn remote_callback(&mut self, f: ~fn()) -> ~RemoteCallback {
        ~UvRemoteCallback::new(self.uvio.uv_loop(), f) as ~RemoteCallback
    }

    fn io<'a>(&'a mut self, f: &fn(&'a mut IoFactory)) {
        f(&mut self.uvio as &mut IoFactory)
    }
}

pub struct UvPausibleIdleCallback {
    priv watcher: IdleWatcher,
    priv idle_flag: bool,
    priv closed: bool
}

impl PausibleIdleCallback for UvPausibleIdleCallback {
    #[inline]
    fn start(&mut self, f: ~fn()) {
        do self.watcher.start |_idle_watcher, _status| {
            f();
        };
        self.idle_flag = true;
    }
    #[inline]
    fn pause(&mut self) {
        if self.idle_flag == true {
            self.watcher.stop();
            self.idle_flag = false;
        }
    }
    #[inline]
    fn resume(&mut self) {
        if self.idle_flag == false {
            self.watcher.restart();
            self.idle_flag = true;
        }
    }
    #[inline]
    fn close(&mut self) {
        self.pause();
        if !self.closed {
            self.closed = true;
            self.watcher.close(||{});
        }
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

// The entire point of async is to call into a loop from other threads so it does not need to home.
pub struct UvRemoteCallback {
    // The uv async handle for triggering the callback
    priv async: AsyncWatcher,
    // A flag to tell the callback to exit, set from the dtor. This is
    // almost never contested - only in rare races with the dtor.
    priv exit_flag: Exclusive<bool>
}

impl UvRemoteCallback {
    pub fn new(loop_: &mut Loop, f: ~fn()) -> UvRemoteCallback {
        let exit_flag = Exclusive::new(false);
        let exit_flag_clone = exit_flag.clone();
        let async = do AsyncWatcher::new(loop_) |watcher, status| {
            assert!(status.is_none());

            // The synchronization logic here is subtle. To review,
            // the uv async handle type promises that, after it is
            // triggered the remote callback is definitely called at
            // least once. UvRemoteCallback needs to maintain those
            // semantics while also shutting down cleanly from the
            // dtor. In our case that means that, when the
            // UvRemoteCallback dtor calls `async.send()`, here `f` is
            // always called later.

            // In the dtor both the exit flag is set and the async
            // callback fired under a lock.  Here, before calling `f`,
            // we take the lock and check the flag. Because we are
            // checking the flag before calling `f`, and the flag is
            // set under the same lock as the send, then if the flag
            // is set then we're guaranteed to call `f` after the
            // final send.

            // If the check was done after `f()` then there would be a
            // period between that call and the check where the dtor
            // could be called in the other thread, missing the final
            // callback while still destroying the handle.

            let should_exit = unsafe {
                exit_flag_clone.with_imm(|&should_exit| should_exit)
            };

            f();

            if should_exit {
                watcher.close(||());
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
    fn drop(&mut self) {
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
        do run_in_mt_newsched_task {
            let mut tube = Tube::new();
            let tube_clone = tube.clone();
            let remote_cell = Cell::new_empty();
            do Local::borrow |sched: &mut Scheduler| {
                let tube_clone = tube_clone.clone();
                let tube_clone_cell = Cell::new(tube_clone);
                let remote = do sched.event_loop.remote_callback {
                    // This could be called multiple times
                    if !tube_clone_cell.is_empty() {
                        tube_clone_cell.take().send(1);
                    }
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

/// Helper for a variety of simple uv_fs_* functions that
/// have no ret val
fn uv_fs_helper(loop_: &mut Loop, path: &CString,
                cb: ~fn(&mut FsRequest, &mut Loop, &CString,
                        ~fn(&FsRequest, Option<UvError>)))
        -> Result<(), IoError> {
    let result_cell = Cell::new_empty();
    let result_cell_ptr: *Cell<Result<(), IoError>> = &result_cell;
    let path_cell = Cell::new(path);
    do task::unkillable { // FIXME(#8674)
        let scheduler: ~Scheduler = Local::take();
        let mut new_req = FsRequest::new();
        do scheduler.deschedule_running_task_and_then |_, task| {
            let task_cell = Cell::new(task);
            let path = path_cell.take();
            do cb(&mut new_req, loop_, path) |_, err| {
                let res = match err {
                    None => Ok(()),
                    Some(err) => Err(uv_error_to_io_error(err))
                };
                unsafe { (*result_cell_ptr).put_back(res); }
                let scheduler: ~Scheduler = Local::take();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            };
        }
    }
    assert!(!result_cell.is_empty());
    return result_cell.take();
}

impl IoFactory for UvIoFactory {
    // Connect to an address and return a new stream
    // NB: This blocks the task waiting on the connection.
    // It would probably be better to return a future
    fn tcp_connect(&mut self, addr: SocketAddr) -> Result<~RtioTcpStream, IoError> {
        // Create a cell in the task to hold the result. We will fill
        // the cell before resuming the task.
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<~RtioTcpStream, IoError>> = &result_cell;

        // Block this task and take ownership, switch to scheduler context
        do task::unkillable { // FIXME(#8674)
            let scheduler: ~Scheduler = Local::take();
            do scheduler.deschedule_running_task_and_then |_, task| {

                let mut tcp = TcpWatcher::new(self.uv_loop());
                let task_cell = Cell::new(task);

                // Wait for a connection
                do tcp.connect(addr) |stream, status| {
                    match status {
                        None => {
                            let tcp = NativeHandle::from_native_handle(stream.native_handle());
                            let home = get_handle_to_current_scheduler!();
                            let res = Ok(~UvTcpStream { watcher: tcp, home: home }
                                                as ~RtioTcpStream);

                            // Store the stream in the task's stack
                            unsafe { (*result_cell_ptr).put_back(res); }

                            // Context switch
                            let scheduler: ~Scheduler = Local::take();
                            scheduler.resume_blocked_task_immediately(task_cell.take());
                        }
                        Some(_) => {
                            let task_cell = Cell::new(task_cell.take());
                            do stream.close {
                                let res = Err(uv_error_to_io_error(status.unwrap()));
                                unsafe { (*result_cell_ptr).put_back(res); }
                                let scheduler: ~Scheduler = Local::take();
                                scheduler.resume_blocked_task_immediately(task_cell.take());
                            }
                        }
                    }
                }
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn tcp_bind(&mut self, addr: SocketAddr) -> Result<~RtioTcpListener, IoError> {
        let mut watcher = TcpWatcher::new(self.uv_loop());
        match watcher.bind(addr) {
            Ok(_) => {
                let home = get_handle_to_current_scheduler!();
                Ok(~UvTcpListener::new(watcher, home) as ~RtioTcpListener)
            }
            Err(uverr) => {
                do task::unkillable { // FIXME(#8674)
                    let scheduler: ~Scheduler = Local::take();
                    do scheduler.deschedule_running_task_and_then |_, task| {
                        let task_cell = Cell::new(task);
                        do watcher.as_stream().close {
                            let scheduler: ~Scheduler = Local::take();
                            scheduler.resume_blocked_task_immediately(task_cell.take());
                        }
                    }
                    Err(uv_error_to_io_error(uverr))
                }
            }
        }
    }

    fn udp_bind(&mut self, addr: SocketAddr) -> Result<~RtioUdpSocket, IoError> {
        let mut watcher = UdpWatcher::new(self.uv_loop());
        match watcher.bind(addr) {
            Ok(_) => {
                let home = get_handle_to_current_scheduler!();
                Ok(~UvUdpSocket { watcher: watcher, home: home } as ~RtioUdpSocket)
            }
            Err(uverr) => {
                do task::unkillable { // FIXME(#8674)
                    let scheduler: ~Scheduler = Local::take();
                    do scheduler.deschedule_running_task_and_then |_, task| {
                        let task_cell = Cell::new(task);
                        do watcher.close {
                            let scheduler: ~Scheduler = Local::take();
                            scheduler.resume_blocked_task_immediately(task_cell.take());
                        }
                    }
                    Err(uv_error_to_io_error(uverr))
                }
            }
        }
    }

    fn timer_init(&mut self) -> Result<~RtioTimer, IoError> {
        let watcher = TimerWatcher::new(self.uv_loop());
        let home = get_handle_to_current_scheduler!();
        Ok(~UvTimer::new(watcher, home) as ~RtioTimer)
    }

    fn fs_from_raw_fd(&mut self, fd: c_int, close: CloseBehavior) -> ~RtioFileStream {
        let loop_ = Loop {handle: self.uv_loop().native_handle()};
        let home = get_handle_to_current_scheduler!();
        ~UvFileStream::new(loop_, fd, close, home) as ~RtioFileStream
    }

    fn fs_open(&mut self, path: &CString, fm: FileMode, fa: FileAccess)
        -> Result<~RtioFileStream, IoError> {
        let mut flags = match fm {
            Open => 0,
            Create => O_CREAT,
            OpenOrCreate => O_CREAT,
            Append => O_APPEND,
            Truncate => O_TRUNC,
            CreateOrTruncate => O_TRUNC | O_CREAT
        };
        flags = match fa {
            Read => flags | O_RDONLY,
            Write => flags | O_WRONLY,
            ReadWrite => flags | O_RDWR
        };
        let create_mode = match fm {
            Create|OpenOrCreate|CreateOrTruncate =>
                S_IRUSR | S_IWUSR,
            _ => 0
        };
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<~RtioFileStream,
                                           IoError>> = &result_cell;
        let path_cell = Cell::new(path);
        do task::unkillable { // FIXME(#8674)
            let scheduler: ~Scheduler = Local::take();
            let open_req = file::FsRequest::new();
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                let path = path_cell.take();
                do open_req.open(self.uv_loop(), path, flags as int, create_mode as int)
                      |req,err| {
                    if err.is_none() {
                        let loop_ = Loop {handle: req.get_loop().native_handle()};
                        let home = get_handle_to_current_scheduler!();
                        let fd = req.get_result() as c_int;
                        let fs = ~UvFileStream::new(
                            loop_, fd, CloseSynchronously, home) as ~RtioFileStream;
                        let res = Ok(fs);
                        unsafe { (*result_cell_ptr).put_back(res); }
                        let scheduler: ~Scheduler = Local::take();
                        scheduler.resume_blocked_task_immediately(task_cell.take());
                    } else {
                        let res = Err(uv_error_to_io_error(err.unwrap()));
                        unsafe { (*result_cell_ptr).put_back(res); }
                        let scheduler: ~Scheduler = Local::take();
                        scheduler.resume_blocked_task_immediately(task_cell.take());
                    }
                };
            };
        };
        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn fs_unlink(&mut self, path: &CString) -> Result<(), IoError> {
        do uv_fs_helper(self.uv_loop(), path) |unlink_req, l, p, cb| {
            do unlink_req.unlink(l, p) |req, err| {
                cb(req, err)
            };
        }
    }
    fn fs_stat(&mut self, path: &CString) -> Result<FileStat, IoError> {
        use str::StrSlice;
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<FileStat,
                                           IoError>> = &result_cell;
        let path_cell = Cell::new(path);
        do task::unkillable { // FIXME(#8674)
            let scheduler: ~Scheduler = Local::take();
            let stat_req = file::FsRequest::new();
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                let path = path_cell.take();
                // Don't pick up the null byte
                let slice = path.as_bytes().slice(0, path.len());
                let path_instance = Cell::new(Path::new(slice));
                do stat_req.stat(self.uv_loop(), path) |req,err| {
                    let res = match err {
                        None => {
                            let stat = req.get_stat();
                            Ok(FileStat {
                                path: path_instance.take(),
                                is_file: stat.is_file(),
                                is_dir: stat.is_dir(),
                                device: stat.st_dev,
                                mode: stat.st_mode,
                                inode: stat.st_ino,
                                size: stat.st_size,
                                created: stat.st_ctim.tv_sec as u64,
                                modified: stat.st_mtim.tv_sec as u64,
                                accessed: stat.st_atim.tv_sec as u64
                            })
                        },
                        Some(e) => {
                            Err(uv_error_to_io_error(e))
                        }
                    };
                    unsafe { (*result_cell_ptr).put_back(res); }
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                };
            };
        };
        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn get_host_addresses(&mut self, host: Option<&str>, servname: Option<&str>,
                          hint: Option<ai::Hint>) -> Result<~[ai::Info], IoError> {
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<~[ai::Info], IoError>> = &result_cell;
        let host_ptr: *Option<&str> = &host;
        let servname_ptr: *Option<&str> = &servname;
        let hint_ptr: *Option<ai::Hint> = &hint;
        let addrinfo_req = GetAddrInfoRequest::new();
        let addrinfo_req_cell = Cell::new(addrinfo_req);

        do task::unkillable { // FIXME(#8674)
            let scheduler: ~Scheduler = Local::take();
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                let mut addrinfo_req = addrinfo_req_cell.take();
                unsafe {
                    do addrinfo_req.getaddrinfo(self.uv_loop(),
                                                *host_ptr, *servname_ptr,
                                                *hint_ptr) |_, addrinfo, err| {
                        let res = match err {
                            None => Ok(accum_addrinfo(addrinfo)),
                            Some(err) => Err(uv_error_to_io_error(err))
                        };
                        (*result_cell_ptr).put_back(res);
                        let scheduler: ~Scheduler = Local::take();
                        scheduler.resume_blocked_task_immediately(task_cell.take());
                    }
                }
            }
        }
        addrinfo_req.delete();
        assert!(!result_cell.is_empty());
        return result_cell.take();
    }
    fn fs_mkdir(&mut self, path: &CString) -> Result<(), IoError> {
        let mode = S_IRWXU as int;
        do uv_fs_helper(self.uv_loop(), path) |mkdir_req, l, p, cb| {
            do mkdir_req.mkdir(l, p, mode as int) |req, err| {
                cb(req, err)
            };
        }
    }
    fn fs_rmdir(&mut self, path: &CString) -> Result<(), IoError> {
        do uv_fs_helper(self.uv_loop(), path) |rmdir_req, l, p, cb| {
            do rmdir_req.rmdir(l, p) |req, err| {
                cb(req, err)
            };
        }
    }
    fn fs_readdir(&mut self, path: &CString, flags: c_int) ->
        Result<~[Path], IoError> {
        use str::StrSlice;
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<~[Path],
                                           IoError>> = &result_cell;
        let path_cell = Cell::new(path);
        do task::unkillable { // FIXME(#8674)
            let scheduler: ~Scheduler = Local::take();
            let stat_req = file::FsRequest::new();
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                let path = path_cell.take();
                // Don't pick up the null byte
                let slice = path.as_bytes().slice(0, path.len());
                let path_parent = Cell::new(Path::new(slice));
                do stat_req.readdir(self.uv_loop(), path, flags) |req,err| {
                    let parent = path_parent.take();
                    let res = match err {
                        None => {
                            let mut paths = ~[];
                            do req.each_path |rel_path| {
                                let p = rel_path.as_bytes();
                                paths.push(parent.join(p.slice_to(rel_path.len())));
                            }
                            Ok(paths)
                        },
                        Some(e) => {
                            Err(uv_error_to_io_error(e))
                        }
                    };
                    unsafe { (*result_cell_ptr).put_back(res); }
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                };
            };
        };
        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn spawn(&mut self, config: ProcessConfig)
            -> Result<(~RtioProcess, ~[Option<~RtioPipe>]), IoError>
    {
        // Sadly, we must create the UvProcess before we actually call uv_spawn
        // so that the exit_cb can close over it and notify it when the process
        // has exited.
        let mut ret = ~UvProcess {
            process: Process::new(),
            home: None,
            exit_status: None,
            term_signal: None,
            exit_error: None,
            descheduled: None,
        };
        let ret_ptr = unsafe {
            *cast::transmute::<&~UvProcess, &*mut UvProcess>(&ret)
        };

        // The purpose of this exit callback is to record the data about the
        // exit and then wake up the task which may be waiting for the process
        // to exit. This is all performed in the current io-loop, and the
        // implementation of UvProcess ensures that reading these fields always
        // occurs on the current io-loop.
        let exit_cb: ExitCallback = |_, exit_status, term_signal, error| {
            unsafe {
                assert!((*ret_ptr).exit_status.is_none());
                (*ret_ptr).exit_status = Some(exit_status);
                (*ret_ptr).term_signal = Some(term_signal);
                (*ret_ptr).exit_error = error;
                match (*ret_ptr).descheduled.take() {
                    Some(task) => {
                        let scheduler: ~Scheduler = Local::take();
                        scheduler.resume_blocked_task_immediately(task);
                    }
                    None => {}
                }
            }
        };

        match ret.process.spawn(self.uv_loop(), config, exit_cb) {
            Ok(io) => {
                // Only now do we actually get a handle to this scheduler.
                ret.home = Some(get_handle_to_current_scheduler!());
                Ok((ret as ~RtioProcess,
                    io.move_iter().map(|p| p.map(|p| p as ~RtioPipe)).collect()))
            }
            Err(uverr) => {
                // We still need to close the process handle we created, but
                // that's taken care for us in the destructor of UvProcess
                Err(uv_error_to_io_error(uverr))
            }
        }
    }

    fn unix_bind(&mut self, path: &CString) ->
        Result<~RtioUnixListener, IoError> {
        let mut pipe = UvUnboundPipe::new(self.uv_loop());
        match pipe.pipe.bind(path) {
            Ok(()) => Ok(~UvUnixListener::new(pipe) as ~RtioUnixListener),
            Err(e) => Err(uv_error_to_io_error(e)),
        }
    }

    fn unix_connect(&mut self, path: &CString) -> Result<~RtioPipe, IoError> {
        let pipe = UvUnboundPipe::new(self.uv_loop());
        let mut rawpipe = pipe.pipe;

        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<~RtioPipe, IoError>> = &result_cell;
        let pipe_cell = Cell::new(pipe);
        let pipe_cell_ptr: *Cell<UvUnboundPipe> = &pipe_cell;

        let scheduler: ~Scheduler = Local::take();
        do scheduler.deschedule_running_task_and_then |_, task| {
            let task_cell = Cell::new(task);
            do rawpipe.connect(path) |_stream, err| {
                let res = match err {
                    None => {
                        let pipe = unsafe { (*pipe_cell_ptr).take() };
                        Ok(~UvPipeStream::new(pipe) as ~RtioPipe)
                    }
                    Some(e) => Err(uv_error_to_io_error(e)),
                };
                unsafe { (*result_cell_ptr).put_back(res); }
                let scheduler: ~Scheduler = Local::take();
                scheduler.resume_blocked_task_immediately(task_cell.take());
            }
        }

        assert!(!result_cell.is_empty());
        return result_cell.take();
    }

    fn tty_open(&mut self, fd: c_int, readable: bool)
            -> Result<~RtioTTY, IoError> {
        match tty::TTY::new(self.uv_loop(), fd, readable) {
            Ok(tty) => Ok(~UvTTY {
                home: get_handle_to_current_scheduler!(),
                tty: tty,
                fd: fd,
            } as ~RtioTTY),
            Err(e) => Err(uv_error_to_io_error(e))
        }
    }

    fn pipe_open(&mut self, fd: c_int) -> Result<~RtioPipe, IoError> {
        let mut pipe = UvUnboundPipe::new(self.uv_loop());
        match pipe.pipe.open(fd) {
            Ok(()) => Ok(~UvPipeStream::new(pipe) as ~RtioPipe),
            Err(e) => Err(uv_error_to_io_error(e))
        }
    }

    fn signal(&mut self, signum: Signum, channel: SharedChan<Signum>)
        -> Result<~RtioSignal, IoError> {
        let watcher = SignalWatcher::new(self.uv_loop());
        let home = get_handle_to_current_scheduler!();
        let mut signal = ~UvSignal::new(watcher, home);
        match signal.watcher.start(signum, |_, _| channel.send_deferred(signum)) {
            Ok(()) => Ok(signal as ~RtioSignal),
            Err(e) => Err(uv_error_to_io_error(e)),
        }
    }
}

pub struct UvTcpListener {
    priv watcher : TcpWatcher,
    priv home: SchedHandle,
}

impl HomingIO for UvTcpListener {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl UvTcpListener {
    fn new(watcher: TcpWatcher, home: SchedHandle) -> UvTcpListener {
        UvTcpListener { watcher: watcher, home: home }
    }
}

impl Drop for UvTcpListener {
    fn drop(&mut self) {
        do self.home_for_io_with_sched |self_, scheduler| {
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task = Cell::new(task);
                do self_.watcher.as_stream().close {
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task.take());
                }
            }
        }
    }
}

impl RtioSocket for UvTcpListener {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        do self.home_for_io |self_| {
            socket_name(Tcp, self_.watcher)
        }
    }
}

impl RtioTcpListener for UvTcpListener {
    fn listen(~self) -> Result<~RtioTcpAcceptor, IoError> {
        do self.home_for_io_consume |self_| {
            let acceptor = ~UvTcpAcceptor::new(self_);
            let incoming = Cell::new(acceptor.incoming.clone());
            let mut stream = acceptor.listener.watcher.as_stream();
            let res = do stream.listen |mut server, status| {
                do incoming.with_mut_ref |incoming| {
                    let inc = match status {
                        Some(_) => Err(standard_error(OtherIoError)),
                        None => {
                            let inc = TcpWatcher::new(&server.event_loop());
                            // first accept call in the callback guarenteed to succeed
                            server.accept(inc.as_stream());
                            let home = get_handle_to_current_scheduler!();
                            Ok(~UvTcpStream { watcher: inc, home: home }
                                    as ~RtioTcpStream)
                        }
                    };
                    incoming.send(inc);
                }
            };
            match res {
                Ok(()) => Ok(acceptor as ~RtioTcpAcceptor),
                Err(e) => Err(uv_error_to_io_error(e)),
            }
        }
    }
}

pub struct UvTcpAcceptor {
    priv listener: UvTcpListener,
    priv incoming: Tube<Result<~RtioTcpStream, IoError>>,
}

impl HomingIO for UvTcpAcceptor {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { self.listener.home() }
}

impl UvTcpAcceptor {
    fn new(listener: UvTcpListener) -> UvTcpAcceptor {
        UvTcpAcceptor { listener: listener, incoming: Tube::new() }
    }
}

impl RtioSocket for UvTcpAcceptor {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        do self.home_for_io |self_| {
            socket_name(Tcp, self_.listener.watcher)
        }
    }
}

fn accept_simultaneously(stream: StreamWatcher, a: int) -> Result<(), IoError> {
    let r = unsafe {
        uvll::tcp_simultaneous_accepts(stream.native_handle(), a as c_int)
    };

    match status_to_maybe_uv_error(r) {
        Some(err) => Err(uv_error_to_io_error(err)),
        None => Ok(())
    }
}

impl RtioTcpAcceptor for UvTcpAcceptor {
    fn accept(&mut self) -> Result<~RtioTcpStream, IoError> {
        do self.home_for_io |self_| {
            self_.incoming.recv()
        }
    }

    fn accept_simultaneously(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            accept_simultaneously(self_.listener.watcher.as_stream(), 1)
        }
    }

    fn dont_accept_simultaneously(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            accept_simultaneously(self_.listener.watcher.as_stream(), 0)
        }
    }
}

fn read_stream(mut watcher: StreamWatcher,
               scheduler: ~Scheduler,
               buf: &mut [u8]) -> Result<uint, IoError> {
    let result_cell = Cell::new_empty();
    let result_cell_ptr: *Cell<Result<uint, IoError>> = &result_cell;

    let buf_ptr: *&mut [u8] = &buf;
    do scheduler.deschedule_running_task_and_then |_sched, task| {
        let task_cell = Cell::new(task);
        // XXX: We shouldn't reallocate these callbacks every
        // call to read
        let alloc: AllocCallback = |_| unsafe {
            slice_to_uv_buf(*buf_ptr)
        };
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

            let scheduler: ~Scheduler = Local::take();
            scheduler.resume_blocked_task_immediately(task_cell.take());
        }
    }

    assert!(!result_cell.is_empty());
    result_cell.take()
}

fn write_stream(mut watcher: StreamWatcher,
                scheduler: ~Scheduler,
                buf: &[u8]) -> Result<(), IoError> {
    let result_cell = Cell::new_empty();
    let result_cell_ptr: *Cell<Result<(), IoError>> = &result_cell;
    let buf_ptr: *&[u8] = &buf;
    do scheduler.deschedule_running_task_and_then |_, task| {
        let task_cell = Cell::new(task);
        let buf = unsafe { slice_to_uv_buf(*buf_ptr) };
        do watcher.write(buf) |_watcher, status| {
            let result = if status.is_none() {
                Ok(())
            } else {
                Err(uv_error_to_io_error(status.unwrap()))
            };

            unsafe { (*result_cell_ptr).put_back(result); }

            let scheduler: ~Scheduler = Local::take();
            scheduler.resume_blocked_task_immediately(task_cell.take());
        }
    }

    assert!(!result_cell.is_empty());
    result_cell.take()
}

pub struct UvUnboundPipe {
    pipe: Pipe,
    priv home: SchedHandle,
}

impl UvUnboundPipe {
    /// Creates a new unbound pipe homed to the current scheduler, placed on the
    /// specified event loop
    pub fn new(loop_: &Loop) -> UvUnboundPipe {
        UvUnboundPipe {
            pipe: Pipe::new(loop_, false),
            home: get_handle_to_current_scheduler!(),
        }
    }
}

impl HomingIO for UvUnboundPipe {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl Drop for UvUnboundPipe {
    fn drop(&mut self) {
        do self.home_for_io |self_| {
            let scheduler: ~Scheduler = Local::take();
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                do self_.pipe.close {
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }
        }
    }
}

pub struct UvPipeStream {
    priv inner: UvUnboundPipe,
}

impl UvPipeStream {
    pub fn new(inner: UvUnboundPipe) -> UvPipeStream {
        UvPipeStream { inner: inner }
    }
}

impl RtioPipe for UvPipeStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        do self.inner.home_for_io_with_sched |self_, scheduler| {
            read_stream(self_.pipe.as_stream(), scheduler, buf)
        }
    }
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        do self.inner.home_for_io_with_sched |self_, scheduler| {
            write_stream(self_.pipe.as_stream(), scheduler, buf)
        }
    }
}

pub struct UvTcpStream {
    priv watcher: TcpWatcher,
    priv home: SchedHandle,
}

impl HomingIO for UvTcpStream {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl Drop for UvTcpStream {
    fn drop(&mut self) {
        do self.home_for_io_with_sched |self_, scheduler| {
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                do self_.watcher.as_stream().close {
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }
        }
    }
}

impl RtioSocket for UvTcpStream {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        do self.home_for_io |self_| {
            socket_name(Tcp, self_.watcher)
        }
    }
}

impl RtioTcpStream for UvTcpStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        do self.home_for_io_with_sched |self_, scheduler| {
            read_stream(self_.watcher.as_stream(), scheduler, buf)
        }
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        do self.home_for_io_with_sched |self_, scheduler| {
            write_stream(self_.watcher.as_stream(), scheduler, buf)
        }
    }

    fn peer_name(&mut self) -> Result<SocketAddr, IoError> {
        do self.home_for_io |self_| {
            socket_name(TcpPeer, self_.watcher)
        }
    }

    fn control_congestion(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            let r = unsafe { uvll::tcp_nodelay(self_.watcher.native_handle(), 0 as c_int) };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn nodelay(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            let r = unsafe { uvll::tcp_nodelay(self_.watcher.native_handle(), 1 as c_int) };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn keepalive(&mut self, delay_in_seconds: uint) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            let r = unsafe {
                uvll::tcp_keepalive(self_.watcher.native_handle(), 1 as c_int,
                                    delay_in_seconds as c_uint)
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn letdie(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            let r = unsafe {
                uvll::tcp_keepalive(self_.watcher.native_handle(), 0 as c_int, 0 as c_uint)
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }
}

pub struct UvUdpSocket {
    priv watcher: UdpWatcher,
    priv home: SchedHandle,
}

impl HomingIO for UvUdpSocket {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl Drop for UvUdpSocket {
    fn drop(&mut self) {
        do self.home_for_io_with_sched |self_, scheduler| {
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                do self_.watcher.close {
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }
        }
    }
}

impl RtioSocket for UvUdpSocket {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        do self.home_for_io |self_| {
            socket_name(Udp, self_.watcher)
        }
    }
}

impl RtioUdpSocket for UvUdpSocket {
    fn recvfrom(&mut self, buf: &mut [u8]) -> Result<(uint, SocketAddr), IoError> {
        do self.home_for_io_with_sched |self_, scheduler| {
            let result_cell = Cell::new_empty();
            let result_cell_ptr: *Cell<Result<(uint, SocketAddr), IoError>> = &result_cell;

            let buf_ptr: *&mut [u8] = &buf;
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                let alloc: AllocCallback = |_| unsafe { slice_to_uv_buf(*buf_ptr) };
                do self_.watcher.recv_start(alloc) |mut watcher, nread, _buf, addr, flags, status| {
                    let _ = flags; // /XXX add handling for partials?

                    watcher.recv_stop();

                    let result = match status {
                        None => {
                            assert!(nread >= 0);
                            Ok((nread as uint, addr))
                        }
                        Some(err) => Err(uv_error_to_io_error(err)),
                    };

                    unsafe { (*result_cell_ptr).put_back(result); }

                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }

            assert!(!result_cell.is_empty());
            result_cell.take()
        }
    }

    fn sendto(&mut self, buf: &[u8], dst: SocketAddr) -> Result<(), IoError> {
        do self.home_for_io_with_sched |self_, scheduler| {
            let result_cell = Cell::new_empty();
            let result_cell_ptr: *Cell<Result<(), IoError>> = &result_cell;
            let buf_ptr: *&[u8] = &buf;
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                let buf = unsafe { slice_to_uv_buf(*buf_ptr) };
                do self_.watcher.send(buf, dst) |_watcher, status| {

                    let result = match status {
                        None => Ok(()),
                        Some(err) => Err(uv_error_to_io_error(err)),
                    };

                    unsafe { (*result_cell_ptr).put_back(result); }

                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }

            assert!(!result_cell.is_empty());
            result_cell.take()
        }
    }

    fn join_multicast(&mut self, multi: IpAddr) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            let r = unsafe {
                do multi.to_str().with_c_str |m_addr| {
                    uvll::udp_set_membership(self_.watcher.native_handle(), m_addr,
                                             ptr::null(), uvll::UV_JOIN_GROUP)
                }
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn leave_multicast(&mut self, multi: IpAddr) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            let r = unsafe {
                do multi.to_str().with_c_str |m_addr| {
                    uvll::udp_set_membership(self_.watcher.native_handle(), m_addr,
                                             ptr::null(), uvll::UV_LEAVE_GROUP)
                }
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn loop_multicast_locally(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {

            let r = unsafe {
                uvll::udp_set_multicast_loop(self_.watcher.native_handle(), 1 as c_int)
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn dont_loop_multicast_locally(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {

            let r = unsafe {
                uvll::udp_set_multicast_loop(self_.watcher.native_handle(), 0 as c_int)
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn multicast_time_to_live(&mut self, ttl: int) -> Result<(), IoError> {
        do self.home_for_io |self_| {

            let r = unsafe {
                uvll::udp_set_multicast_ttl(self_.watcher.native_handle(), ttl as c_int)
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn time_to_live(&mut self, ttl: int) -> Result<(), IoError> {
        do self.home_for_io |self_| {

            let r = unsafe {
                uvll::udp_set_ttl(self_.watcher.native_handle(), ttl as c_int)
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn hear_broadcasts(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {

            let r = unsafe {
                uvll::udp_set_broadcast(self_.watcher.native_handle(), 1 as c_int)
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }

    fn ignore_broadcasts(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {

            let r = unsafe {
                uvll::udp_set_broadcast(self_.watcher.native_handle(), 0 as c_int)
            };

            match status_to_maybe_uv_error(r) {
                Some(err) => Err(uv_error_to_io_error(err)),
                None => Ok(())
            }
        }
    }
}

pub struct UvTimer {
    priv watcher: timer::TimerWatcher,
    priv home: SchedHandle,
}

impl HomingIO for UvTimer {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl UvTimer {
    fn new(w: timer::TimerWatcher, home: SchedHandle) -> UvTimer {
        UvTimer { watcher: w, home: home }
    }
}

impl Drop for UvTimer {
    fn drop(&mut self) {
        do self.home_for_io_with_sched |self_, scheduler| {
            rtdebug!("closing UvTimer");
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                do self_.watcher.close {
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }
        }
    }
}

impl RtioTimer for UvTimer {
    fn sleep(&mut self, msecs: u64) {
        do self.home_for_io_with_sched |self_, scheduler| {
            do scheduler.deschedule_running_task_and_then |_sched, task| {
                rtdebug!("sleep: entered scheduler context");
                let task_cell = Cell::new(task);
                do self_.watcher.start(msecs, 0) |_, status| {
                    assert!(status.is_none());
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }
            self_.watcher.stop();
        }
    }
}

pub struct UvFileStream {
    priv loop_: Loop,
    priv fd: c_int,
    priv close: CloseBehavior,
    priv home: SchedHandle,
}

impl HomingIO for UvFileStream {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl UvFileStream {
    fn new(loop_: Loop, fd: c_int, close: CloseBehavior,
           home: SchedHandle) -> UvFileStream {
        UvFileStream {
            loop_: loop_,
            fd: fd,
            close: close,
            home: home,
        }
    }
    fn base_read(&mut self, buf: &mut [u8], offset: i64) -> Result<int, IoError> {
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<int, IoError>> = &result_cell;
        let buf_ptr: *&mut [u8] = &buf;
        do self.home_for_io_with_sched |self_, scheduler| {
            do scheduler.deschedule_running_task_and_then |_, task| {
                let buf = unsafe { slice_to_uv_buf(*buf_ptr) };
                let task_cell = Cell::new(task);
                let read_req = file::FsRequest::new();
                do read_req.read(&self_.loop_, self_.fd, buf, offset) |req, uverr| {
                    let res = match uverr  {
                        None => Ok(req.get_result() as int),
                        Some(err) => Err(uv_error_to_io_error(err))
                    };
                    unsafe { (*result_cell_ptr).put_back(res); }
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }
        }
        result_cell.take()
    }
    fn base_write(&mut self, buf: &[u8], offset: i64) -> Result<(), IoError> {
        let result_cell = Cell::new_empty();
        let result_cell_ptr: *Cell<Result<(), IoError>> = &result_cell;
        let buf_ptr: *&[u8] = &buf;
        do self.home_for_io_with_sched |self_, scheduler| {
            do scheduler.deschedule_running_task_and_then |_, task| {
                let buf = unsafe { slice_to_uv_buf(*buf_ptr) };
                let task_cell = Cell::new(task);
                let write_req = file::FsRequest::new();
                do write_req.write(&self_.loop_, self_.fd, buf, offset) |_, uverr| {
                    let res = match uverr  {
                        None => Ok(()),
                        Some(err) => Err(uv_error_to_io_error(err))
                    };
                    unsafe { (*result_cell_ptr).put_back(res); }
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }
        }
        result_cell.take()
    }
    fn seek_common(&mut self, pos: i64, whence: c_int) ->
        Result<u64, IoError>{
        #[fixed_stack_segment]; #[inline(never)];
        unsafe {
            match lseek(self.fd, pos as off_t, whence) {
                -1 => {
                    Err(IoError {
                        kind: OtherIoError,
                        desc: "Failed to lseek.",
                        detail: None
                    })
                },
                n => Ok(n as u64)
            }
        }
    }
}

impl Drop for UvFileStream {
    fn drop(&mut self) {
        match self.close {
            DontClose => {}
            CloseAsynchronously => {
                let close_req = file::FsRequest::new();
                do close_req.close(&self.loop_, self.fd) |_,_| {}
            }
            CloseSynchronously => {
                do self.home_for_io_with_sched |self_, scheduler| {
                    do scheduler.deschedule_running_task_and_then |_, task| {
                        let task_cell = Cell::new(task);
                        let close_req = file::FsRequest::new();
                        do close_req.close(&self_.loop_, self_.fd) |_,_| {
                            let scheduler: ~Scheduler = Local::take();
                            scheduler.resume_blocked_task_immediately(task_cell.take());
                        }
                    }
                }
            }
        }
    }
}

impl RtioFileStream for UvFileStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<int, IoError> {
        self.base_read(buf, -1)
    }
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        self.base_write(buf, -1)
    }
    fn pread(&mut self, buf: &mut [u8], offset: u64) -> Result<int, IoError> {
        self.base_read(buf, offset as i64)
    }
    fn pwrite(&mut self, buf: &[u8], offset: u64) -> Result<(), IoError> {
        self.base_write(buf, offset as i64)
    }
    fn seek(&mut self, pos: i64, whence: SeekStyle) -> Result<u64, IoError> {
        use libc::{SEEK_SET, SEEK_CUR, SEEK_END};
        let whence = match whence {
            SeekSet => SEEK_SET,
            SeekCur => SEEK_CUR,
            SeekEnd => SEEK_END
        };
        self.seek_common(pos, whence)
    }
    fn tell(&self) -> Result<u64, IoError> {
        use libc::SEEK_CUR;
        // this is temporary
        let self_ = unsafe { cast::transmute::<&UvFileStream, &mut UvFileStream>(self) };
        self_.seek_common(0, SEEK_CUR)
    }
    fn flush(&mut self) -> Result<(), IoError> {
        Ok(())
    }
}

pub struct UvProcess {
    priv process: process::Process,

    // Sadly, this structure must be created before we return it, so in that
    // brief interim the `home` is None.
    priv home: Option<SchedHandle>,

    // All None until the process exits (exit_error may stay None)
    priv exit_status: Option<int>,
    priv term_signal: Option<int>,
    priv exit_error: Option<UvError>,

    // Used to store which task to wake up from the exit_cb
    priv descheduled: Option<BlockedTask>,
}

impl HomingIO for UvProcess {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { self.home.get_mut_ref() }
}

impl Drop for UvProcess {
    fn drop(&mut self) {
        let close = |self_: &mut UvProcess| {
            let scheduler: ~Scheduler = Local::take();
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task = Cell::new(task);
                do self_.process.close {
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task.take());
                }
            }
        };

        // If home is none, then this process never actually successfully
        // spawned, so there's no need to switch event loops
        if self.home.is_none() {
            close(self)
        } else {
            self.home_for_io(close)
        }
    }
}

impl RtioProcess for UvProcess {
    fn id(&self) -> pid_t {
        self.process.pid()
    }

    fn kill(&mut self, signal: int) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            match self_.process.kill(signal) {
                Ok(()) => Ok(()),
                Err(uverr) => Err(uv_error_to_io_error(uverr))
            }
        }
    }

    fn wait(&mut self) -> int {
        // Make sure (on the home scheduler) that we have an exit status listed
        do self.home_for_io |self_| {
            match self_.exit_status {
                Some(*) => {}
                None => {
                    // If there's no exit code previously listed, then the
                    // process's exit callback has yet to be invoked. We just
                    // need to deschedule ourselves and wait to be reawoken.
                    let scheduler: ~Scheduler = Local::take();
                    do scheduler.deschedule_running_task_and_then |_, task| {
                        assert!(self_.descheduled.is_none());
                        self_.descheduled = Some(task);
                    }
                    assert!(self_.exit_status.is_some());
                }
            }
        }

        self.exit_status.unwrap()
    }
}

pub struct UvUnixListener {
    priv inner: UvUnboundPipe
}

impl HomingIO for UvUnixListener {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { self.inner.home() }
}

impl UvUnixListener {
    fn new(pipe: UvUnboundPipe) -> UvUnixListener {
        UvUnixListener { inner: pipe }
    }
}

impl RtioUnixListener for UvUnixListener {
    fn listen(~self) -> Result<~RtioUnixAcceptor, IoError> {
        do self.home_for_io_consume |self_| {
            let acceptor = ~UvUnixAcceptor::new(self_);
            let incoming = Cell::new(acceptor.incoming.clone());
            let mut stream = acceptor.listener.inner.pipe.as_stream();
            let res = do stream.listen |mut server, status| {
                do incoming.with_mut_ref |incoming| {
                    let inc = match status {
                        Some(e) => Err(uv_error_to_io_error(e)),
                        None => {
                            let pipe = UvUnboundPipe::new(&server.event_loop());
                            server.accept(pipe.pipe.as_stream());
                            Ok(~UvPipeStream::new(pipe) as ~RtioPipe)
                        }
                    };
                    incoming.send(inc);
                }
            };
            match res {
                Ok(()) => Ok(acceptor as ~RtioUnixAcceptor),
                Err(e) => Err(uv_error_to_io_error(e)),
            }
        }
    }
}

pub struct UvTTY {
    tty: tty::TTY,
    home: SchedHandle,
    fd: c_int,
}

impl HomingIO for UvTTY {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl Drop for UvTTY {
    fn drop(&mut self) {
        // TTY handles are used for the logger in a task, so this destructor is
        // run when a task is destroyed. When a task is being destroyed, a local
        // scheduler isn't available, so we can't do the normal "take the
        // scheduler and resume once close is done". Instead close operations on
        // a TTY are asynchronous.
        self.tty.close_async();
    }
}

impl RtioTTY for UvTTY {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        do self.home_for_io_with_sched |self_, scheduler| {
            read_stream(self_.tty.as_stream(), scheduler, buf)
        }
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        do self.home_for_io_with_sched |self_, scheduler| {
            write_stream(self_.tty.as_stream(), scheduler, buf)
        }
    }

    fn set_raw(&mut self, raw: bool) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            match self_.tty.set_mode(raw) {
                Ok(p) => Ok(p), Err(e) => Err(uv_error_to_io_error(e))
            }
        }
    }

    fn get_winsize(&mut self) -> Result<(int, int), IoError> {
        do self.home_for_io |self_| {
            match self_.tty.get_winsize() {
                Ok(p) => Ok(p), Err(e) => Err(uv_error_to_io_error(e))
            }
        }
    }

    fn isatty(&self) -> bool {
        unsafe { uvll::guess_handle(self.fd) == uvll::UV_TTY as c_int }
    }
}

pub struct UvUnixAcceptor {
    listener: UvUnixListener,
    incoming: Tube<Result<~RtioPipe, IoError>>,
}

impl HomingIO for UvUnixAcceptor {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { self.listener.home() }
}

impl UvUnixAcceptor {
    fn new(listener: UvUnixListener) -> UvUnixAcceptor {
        UvUnixAcceptor { listener: listener, incoming: Tube::new() }
    }
}

impl RtioUnixAcceptor for UvUnixAcceptor {
    fn accept(&mut self) -> Result<~RtioPipe, IoError> {
        do self.home_for_io |self_| {
            self_.incoming.recv()
        }
    }

    fn accept_simultaneously(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            accept_simultaneously(self_.listener.inner.pipe.as_stream(), 1)
        }
    }

    fn dont_accept_simultaneously(&mut self) -> Result<(), IoError> {
        do self.home_for_io |self_| {
            accept_simultaneously(self_.listener.inner.pipe.as_stream(), 0)
        }
    }
}

pub struct UvSignal {
    watcher: signal::SignalWatcher,
    home: SchedHandle,
}

impl HomingIO for UvSignal {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl UvSignal {
    fn new(w: signal::SignalWatcher, home: SchedHandle) -> UvSignal {
        UvSignal { watcher: w, home: home }
    }
}

impl RtioSignal for UvSignal {}

impl Drop for UvSignal {
    fn drop(&mut self) {
        do self.home_for_io_with_sched |self_, scheduler| {
            rtdebug!("closing UvSignal");
            do scheduler.deschedule_running_task_and_then |_, task| {
                let task_cell = Cell::new(task);
                do self_.watcher.close {
                    let scheduler: ~Scheduler = Local::take();
                    scheduler.resume_blocked_task_immediately(task_cell.take());
                }
            }
        }
    }
}

// this function is full of lies
unsafe fn local_io() -> &'static mut IoFactory {
    do Local::borrow |sched: &mut Scheduler| {
        let mut io = None;
        sched.event_loop.io(|i| io = Some(i));
        cast::transmute(io.unwrap())
    }
}

#[test]
fn test_simple_io_no_connect() {
    do run_in_mt_newsched_task {
        unsafe {
            let io = local_io();
            let addr = next_test_ip4();
            let maybe_chan = io.tcp_connect(addr);
            assert!(maybe_chan.is_err());
        }
    }
}

#[test]
fn test_simple_udp_io_bind_only() {
    do run_in_mt_newsched_task {
        unsafe {
            let io = local_io();
            let addr = next_test_ip4();
            let maybe_socket = io.udp_bind(addr);
            assert!(maybe_socket.is_ok());
        }
    }
}

#[test]
fn test_simple_homed_udp_io_bind_then_move_task_then_home_and_close() {
    use rt::sleeper_list::SleeperList;
    use rt::work_queue::WorkQueue;
    use rt::thread::Thread;
    use rt::task::Task;
    use rt::sched::{Shutdown, TaskFromFriend};
    do run_in_bare_thread {
        let sleepers = SleeperList::new();
        let work_queue1 = WorkQueue::new();
        let work_queue2 = WorkQueue::new();
        let queues = ~[work_queue1.clone(), work_queue2.clone()];

        let loop1 = ~UvEventLoop::new() as ~EventLoop;
        let mut sched1 = ~Scheduler::new(loop1, work_queue1, queues.clone(),
                                         sleepers.clone());
        let loop2 = ~UvEventLoop::new() as ~EventLoop;
        let mut sched2 = ~Scheduler::new(loop2, work_queue2, queues.clone(),
                                         sleepers.clone());

        let handle1 = Cell::new(sched1.make_handle());
        let handle2 = Cell::new(sched2.make_handle());
        let tasksFriendHandle = Cell::new(sched2.make_handle());

        let on_exit: ~fn(bool) = |exit_status| {
            handle1.take().send(Shutdown);
            handle2.take().send(Shutdown);
            rtassert!(exit_status);
        };

        let test_function: ~fn() = || {
            let io = unsafe { local_io() };
            let addr = next_test_ip4();
            let maybe_socket = io.udp_bind(addr);
            // this socket is bound to this event loop
            assert!(maybe_socket.is_ok());

            // block self on sched1
            do task::unkillable { // FIXME(#8674)
                let scheduler: ~Scheduler = Local::take();
                do scheduler.deschedule_running_task_and_then |_, task| {
                    // unblock task
                    do task.wake().map |task| {
                      // send self to sched2
                      tasksFriendHandle.take().send(TaskFromFriend(task));
                    };
                    // sched1 should now sleep since it has nothing else to do
                }
            }
            // sched2 will wake up and get the task
            // as we do nothing else, the function ends and the socket goes out of scope
            // sched2 will start to run the destructor
            // the destructor will first block the task, set it's home as sched1, then enqueue it
            // sched2 will dequeue the task, see that it has a home, and send it to sched1
            // sched1 will wake up, exec the close function on the correct loop, and then we're done
        };

        let mut main_task = ~Task::new_root(&mut sched1.stack_pool, None, test_function);
        main_task.death.on_exit = Some(on_exit);
        let main_task = Cell::new(main_task);

        let null_task = Cell::new(~do Task::new_root(&mut sched2.stack_pool, None) || {});

        let sched1 = Cell::new(sched1);
        let sched2 = Cell::new(sched2);

        let thread1 = do Thread::start {
            sched1.take().bootstrap(main_task.take());
        };
        let thread2 = do Thread::start {
            sched2.take().bootstrap(null_task.take());
        };

        thread1.join();
        thread2.join();
    }
}

#[test]
fn test_simple_homed_udp_io_bind_then_move_handle_then_home_and_close() {
    use rt::sleeper_list::SleeperList;
    use rt::work_queue::WorkQueue;
    use rt::thread::Thread;
    use rt::task::Task;
    use rt::comm::oneshot;
    use rt::sched::Shutdown;
    do run_in_bare_thread {
        let sleepers = SleeperList::new();
        let work_queue1 = WorkQueue::new();
        let work_queue2 = WorkQueue::new();
        let queues = ~[work_queue1.clone(), work_queue2.clone()];

        let loop1 = ~UvEventLoop::new() as ~EventLoop;
        let mut sched1 = ~Scheduler::new(loop1, work_queue1, queues.clone(),
                                         sleepers.clone());
        let loop2 = ~UvEventLoop::new() as ~EventLoop;
        let mut sched2 = ~Scheduler::new(loop2, work_queue2, queues.clone(),
                                         sleepers.clone());

        let handle1 = Cell::new(sched1.make_handle());
        let handle2 = Cell::new(sched2.make_handle());

        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        let body1: ~fn() = || {
            let io = unsafe { local_io() };
            let addr = next_test_ip4();
            let socket = io.udp_bind(addr);
            assert!(socket.is_ok());
            chan.take().send(socket);
        };

        let body2: ~fn() = || {
            let socket = port.take().recv();
            assert!(socket.is_ok());
            /* The socket goes out of scope and the destructor is called.
             * The destructor:
             *  - sends itself back to sched1
             *  - frees the socket
             *  - resets the home of the task to whatever it was previously
             */
        };

        let on_exit: ~fn(bool) = |exit| {
            handle1.take().send(Shutdown);
            handle2.take().send(Shutdown);
            rtassert!(exit);
        };

        let task1 = Cell::new(~Task::new_root(&mut sched1.stack_pool, None, body1));

        let mut task2 = ~Task::new_root(&mut sched2.stack_pool, None, body2);
        task2.death.on_exit = Some(on_exit);
        let task2 = Cell::new(task2);

        let sched1 = Cell::new(sched1);
        let sched2 = Cell::new(sched2);

        let thread1 = do Thread::start {
            sched1.take().bootstrap(task1.take());
        };
        let thread2 = do Thread::start {
            sched2.take().bootstrap(task2.take());
        };

        thread1.join();
        thread2.join();
    }
}

#[test]
fn test_simple_tcp_server_and_client() {
    do run_in_mt_newsched_task {
        let addr = next_test_ip4();
        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        // Start the server first so it's listening when we connect
        do spawntask {
            unsafe {
                let io = local_io();
                let listener = io.tcp_bind(addr).unwrap();
                let mut acceptor = listener.listen().unwrap();
                chan.take().send(());
                let mut stream = acceptor.accept().unwrap();
                let mut buf = [0, .. 2048];
                let nread = stream.read(buf).unwrap();
                assert_eq!(nread, 8);
                for i in range(0u, nread) {
                    rtdebug!("{}", buf[i]);
                    assert_eq!(buf[i], i as u8);
                }
            }
        }

        do spawntask {
            unsafe {
                port.take().recv();
                let io = local_io();
                let mut stream = io.tcp_connect(addr).unwrap();
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
            }
        }
    }
}

#[test]
fn test_simple_tcp_server_and_client_on_diff_threads() {
    use rt::sleeper_list::SleeperList;
    use rt::work_queue::WorkQueue;
    use rt::thread::Thread;
    use rt::task::Task;
    use rt::sched::{Shutdown};
    do run_in_bare_thread {
        let sleepers = SleeperList::new();

        let server_addr = next_test_ip4();
        let client_addr = server_addr.clone();

        let server_work_queue = WorkQueue::new();
        let client_work_queue = WorkQueue::new();
        let queues = ~[server_work_queue.clone(), client_work_queue.clone()];

        let sloop = ~UvEventLoop::new() as ~EventLoop;
        let mut server_sched = ~Scheduler::new(sloop, server_work_queue,
                                               queues.clone(), sleepers.clone());
        let cloop = ~UvEventLoop::new() as ~EventLoop;
        let mut client_sched = ~Scheduler::new(cloop, client_work_queue,
                                               queues.clone(), sleepers.clone());

        let server_handle = Cell::new(server_sched.make_handle());
        let client_handle = Cell::new(client_sched.make_handle());

        let server_on_exit: ~fn(bool) = |exit_status| {
            server_handle.take().send(Shutdown);
            rtassert!(exit_status);
        };

        let client_on_exit: ~fn(bool) = |exit_status| {
            client_handle.take().send(Shutdown);
            rtassert!(exit_status);
        };

        let server_fn: ~fn() = || {
            let io = unsafe { local_io() };
            let listener = io.tcp_bind(server_addr).unwrap();
            let mut acceptor = listener.listen().unwrap();
            let mut stream = acceptor.accept().unwrap();
            let mut buf = [0, .. 2048];
            let nread = stream.read(buf).unwrap();
            assert_eq!(nread, 8);
            for i in range(0u, nread) {
                assert_eq!(buf[i], i as u8);
            }
        };

        let client_fn: ~fn() = || {
            let io = unsafe { local_io() };
            let mut stream = io.tcp_connect(client_addr);
            while stream.is_err() {
                stream = io.tcp_connect(client_addr);
            }
            stream.unwrap().write([0, 1, 2, 3, 4, 5, 6, 7]);
        };

        let mut server_task = ~Task::new_root(&mut server_sched.stack_pool, None, server_fn);
        server_task.death.on_exit = Some(server_on_exit);
        let server_task = Cell::new(server_task);

        let mut client_task = ~Task::new_root(&mut client_sched.stack_pool, None, client_fn);
        client_task.death.on_exit = Some(client_on_exit);
        let client_task = Cell::new(client_task);

        let server_sched = Cell::new(server_sched);
        let client_sched = Cell::new(client_sched);

        let server_thread = do Thread::start {
            server_sched.take().bootstrap(server_task.take());
        };
        let client_thread = do Thread::start {
            client_sched.take().bootstrap(client_task.take());
        };

        server_thread.join();
        client_thread.join();
    }
}

#[test]
fn test_simple_udp_server_and_client() {
    do run_in_mt_newsched_task {
        let server_addr = next_test_ip4();
        let client_addr = next_test_ip4();
        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        do spawntask {
            unsafe {
                let io = local_io();
                let mut server_socket = io.udp_bind(server_addr).unwrap();
                chan.take().send(());
                let mut buf = [0, .. 2048];
                let (nread,src) = server_socket.recvfrom(buf).unwrap();
                assert_eq!(nread, 8);
                for i in range(0u, nread) {
                    rtdebug!("{}", buf[i]);
                    assert_eq!(buf[i], i as u8);
                }
                assert_eq!(src, client_addr);
            }
        }

        do spawntask {
            unsafe {
                let io = local_io();
                let mut client_socket = io.udp_bind(client_addr).unwrap();
                port.take().recv();
                client_socket.sendto([0, 1, 2, 3, 4, 5, 6, 7], server_addr);
            }
        }
    }
}

#[test] #[ignore(reason = "busted")]
fn test_read_and_block() {
    do run_in_mt_newsched_task {
        let addr = next_test_ip4();
        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        do spawntask {
            let io = unsafe { local_io() };
            let listener = io.tcp_bind(addr).unwrap();
            let mut acceptor = listener.listen().unwrap();
            chan.take().send(());
            let mut stream = acceptor.accept().unwrap();
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

                do task::unkillable { // FIXME(#8674)
                    let scheduler: ~Scheduler = Local::take();
                    // Yield to the other task in hopes that it
                    // will trigger a read callback while we are
                    // not ready for it
                    do scheduler.deschedule_running_task_and_then |sched, task| {
                        let task = Cell::new(task);
                        sched.enqueue_blocked_task(task.take());
                    }
                }
            }

            // Make sure we had multiple reads
            assert!(reads > 1);
        }

        do spawntask {
            unsafe {
                port.take().recv();
                let io = local_io();
                let mut stream = io.tcp_connect(addr).unwrap();
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
    do run_in_mt_newsched_task {
        let addr = next_test_ip4();
        static MAX: uint = 500000;
        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        do spawntask {
            unsafe {
                let io = local_io();
                let listener = io.tcp_bind(addr).unwrap();
                let mut acceptor = listener.listen().unwrap();
                chan.take().send(());
                let mut stream = acceptor.accept().unwrap();
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
                port.take().recv();
                let io = local_io();
                let mut stream = io.tcp_connect(addr).unwrap();
                let mut buf = [0, .. 2048];
                let mut total_bytes_read = 0;
                while total_bytes_read < MAX {
                    let nread = stream.read(buf).unwrap();
                    rtdebug!("read {} bytes", nread);
                    total_bytes_read += nread;
                    for i in range(0u, nread) {
                        assert_eq!(buf[i], 1);
                    }
                }
                rtdebug!("read {} bytes total", total_bytes_read);
            }
        }
    }
}

#[test]
fn test_udp_twice() {
    do run_in_mt_newsched_task {
        let server_addr = next_test_ip4();
        let client_addr = next_test_ip4();
        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        do spawntask {
            unsafe {
                let io = local_io();
                let mut client = io.udp_bind(client_addr).unwrap();
                port.take().recv();
                assert!(client.sendto([1], server_addr).is_ok());
                assert!(client.sendto([2], server_addr).is_ok());
            }
        }

        do spawntask {
            unsafe {
                let io = local_io();
                let mut server = io.udp_bind(server_addr).unwrap();
                chan.take().send(());
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
    do run_in_mt_newsched_task {
        let server_out_addr = next_test_ip4();
        let server_in_addr = next_test_ip4();
        let client_out_addr = next_test_ip4();
        let client_in_addr = next_test_ip4();
        static MAX: uint = 500_000;

        let (p1, c1) = oneshot();
        let (p2, c2) = oneshot();

        let first = Cell::new((p1, c2));
        let second = Cell::new((p2, c1));

        do spawntask {
            unsafe {
                let io = local_io();
                let mut server_out = io.udp_bind(server_out_addr).unwrap();
                let mut server_in = io.udp_bind(server_in_addr).unwrap();
                let (port, chan) = first.take();
                chan.send(());
                port.recv();
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
                let io = local_io();
                let mut client_out = io.udp_bind(client_out_addr).unwrap();
                let mut client_in = io.udp_bind(client_in_addr).unwrap();
                let (port, chan) = second.take();
                port.recv();
                chan.send(());
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

#[test]
fn test_timer_sleep_simple() {
    do run_in_mt_newsched_task {
        unsafe {
            let io = local_io();
            let timer = io.timer_init();
            do timer.map_move |mut t| { t.sleep(1) };
        }
    }
}

fn file_test_uvio_full_simple_impl() {
    use str::StrSlice; // why does this have to be explicitly imported to work?
                       // compiler was complaining about no trait for str that
                       // does .as_bytes() ..
    use rt::io::{Open, Create, ReadWrite, Read};
    unsafe {
        let io = local_io();
        let write_val = "hello uvio!";
        let path = "./tmp/file_test_uvio_full.txt";
        {
            let create_fm = Create;
            let create_fa = ReadWrite;
            let mut fd = io.fs_open(&path.to_c_str(), create_fm, create_fa).unwrap();
            let write_buf = write_val.as_bytes();
            fd.write(write_buf);
        }
        {
            let ro_fm = Open;
            let ro_fa = Read;
            let mut fd = io.fs_open(&path.to_c_str(), ro_fm, ro_fa).unwrap();
            let mut read_vec = [0, .. 1028];
            let nread = fd.read(read_vec).unwrap();
            let read_val = str::from_utf8(read_vec.slice(0, nread as uint));
            assert!(read_val == write_val.to_owned());
        }
        io.fs_unlink(&path.to_c_str());
    }
}

#[test]
fn file_test_uvio_full_simple() {
    do run_in_mt_newsched_task {
        file_test_uvio_full_simple_impl();
    }
}

fn uvio_naive_print(input: &str) {
    use str::StrSlice;
    unsafe {
        use libc::{STDOUT_FILENO};
        let io = local_io();
        {
            let mut fd = io.fs_from_raw_fd(STDOUT_FILENO, DontClose);
            let write_buf = input.as_bytes();
            fd.write(write_buf);
        }
    }
}

#[test]
fn file_test_uvio_write_to_stdout() {
    do run_in_mt_newsched_task {
        uvio_naive_print("jubilation\n");
    }
}
