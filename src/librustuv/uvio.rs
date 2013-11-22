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
use std::comm::SharedChan;
use std::libc::c_int;
use std::libc;
use std::path::Path;
use std::io::IoError;
use std::io::net::ip::SocketAddr;
use std::io::process::ProcessConfig;
use std::io;
use std::rt::local::Local;
use std::rt::rtio::*;
use std::rt::sched::{Scheduler, SchedHandle};
use std::rt::task::Task;
use std::libc::{O_CREAT, O_APPEND, O_TRUNC, O_RDWR, O_RDONLY, O_WRONLY,
                S_IRUSR, S_IWUSR};
use std::io::{FileMode, FileAccess, Open, Append, Truncate, Read, Write,
                  ReadWrite, FileStat};
use std::io::signal::Signum;
use std::util;
use ai = std::io::net::addrinfo;

#[cfg(test)] use std::unstable::run_in_bare_thread;

use super::*;
use addrinfo::GetAddrInfoRequest;

pub trait HomingIO {

    fn home<'r>(&'r mut self) -> &'r mut SchedHandle;

    /// This function will move tasks to run on their home I/O scheduler. Note
    /// that this function does *not* pin the task to the I/O scheduler, but
    /// rather it simply moves it to running on the I/O scheduler.
    fn go_to_IO_home(&mut self) -> uint {
        use std::rt::sched::RunOnce;

        let _f = ForbidUnwind::new("going home");

        let current_sched_id = do Local::borrow |sched: &mut Scheduler| {
            sched.sched_id()
        };

        // Only need to invoke a context switch if we're not on the right
        // scheduler.
        if current_sched_id != self.home().sched_id {
            let scheduler: ~Scheduler = Local::take();
            do scheduler.deschedule_running_task_and_then |_, task| {
                do task.wake().map |task| {
                    self.home().send(RunOnce(task));
                };
            }
        }
        let current_sched_id = do Local::borrow |sched: &mut Scheduler| {
            sched.sched_id()
        };
        assert!(current_sched_id == self.home().sched_id);

        self.home().sched_id
    }

    /// Fires a single homing missile, returning another missile targeted back
    /// at the original home of this task. In other words, this function will
    /// move the local task to its I/O scheduler and then return an RAII wrapper
    /// which will return the task home.
    fn fire_homing_missile(&mut self) -> HomingMissile {
        HomingMissile { io_home: self.go_to_IO_home() }
    }

    /// Same as `fire_homing_missile`, but returns the local I/O scheduler as
    /// well (the one that was homed to).
    fn fire_homing_missile_sched(&mut self) -> (HomingMissile, ~Scheduler) {
        // First, transplant ourselves to the home I/O scheduler
        let missile = self.fire_homing_missile();
        // Next (must happen next), grab the local I/O scheduler
        let io_sched: ~Scheduler = Local::take();

        (missile, io_sched)
    }
}

/// After a homing operation has been completed, this will return the current
/// task back to its appropriate home (if applicable). The field is used to
/// assert that we are where we think we are.
struct HomingMissile {
    priv io_home: uint,
}

impl HomingMissile {
    pub fn check(&self, msg: &'static str) {
        let local_id = Local::borrow(|sched: &mut Scheduler| sched.sched_id());
        assert!(local_id == self.io_home, "{}", msg);
    }
}

impl Drop for HomingMissile {
    fn drop(&mut self) {
        let f = ForbidUnwind::new("leaving home");

        // It would truly be a sad day if we had moved off the home I/O
        // scheduler while we were doing I/O.
        self.check("task moved away from the home scheduler");

        // If we were a homed task, then we must send ourselves back to the
        // original scheduler. Otherwise, we can just return and keep running
        if !Task::on_appropriate_sched() {
            let scheduler: ~Scheduler = Local::take();
            do scheduler.deschedule_running_task_and_then |_, task| {
                do task.wake().map |task| {
                    Scheduler::run_task(task);
                };
            }
        }

        util::ignore(f);
    }
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

    fn callback(&mut self, f: proc()) {
        IdleWatcher::onetime(self.uvio.uv_loop(), f);
    }

    fn pausible_idle_callback(&mut self, cb: ~Callback) -> ~PausibleIdleCallback {
        IdleWatcher::new(self.uvio.uv_loop(), cb) as ~PausibleIdleCallback
    }

    fn remote_callback(&mut self, f: ~Callback) -> ~RemoteCallback {
        ~AsyncWatcher::new(self.uvio.uv_loop(), f) as ~RemoteCallback
    }

    fn io<'a>(&'a mut self, f: &fn(&'a mut IoFactory)) {
        f(&mut self.uvio as &mut IoFactory)
    }
}

#[cfg(not(test))]
#[lang = "event_loop_factory"]
pub extern "C" fn new_loop() -> ~EventLoop {
    ~UvEventLoop::new() as ~EventLoop
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
    fn tcp_connect(&mut self, addr: SocketAddr)
        -> Result<~RtioTcpStream, IoError>
    {
        match TcpWatcher::connect(self.uv_loop(), addr) {
            Ok(t) => Ok(~t as ~RtioTcpStream),
            Err(e) => Err(uv_error_to_io_error(e)),
        }
    }

    fn tcp_bind(&mut self, addr: SocketAddr) -> Result<~RtioTcpListener, IoError> {
        match TcpListener::bind(self.uv_loop(), addr) {
            Ok(t) => Ok(t as ~RtioTcpListener),
            Err(e) => Err(uv_error_to_io_error(e)),
        }
    }

    fn udp_bind(&mut self, addr: SocketAddr) -> Result<~RtioUdpSocket, IoError> {
        match UdpWatcher::bind(self.uv_loop(), addr) {
            Ok(u) => Ok(~u as ~RtioUdpSocket),
            Err(e) => Err(uv_error_to_io_error(e)),
        }
    }

    fn timer_init(&mut self) -> Result<~RtioTimer, IoError> {
        Ok(TimerWatcher::new(self.uv_loop()) as ~RtioTimer)
    }

    fn get_host_addresses(&mut self, host: Option<&str>, servname: Option<&str>,
                          hint: Option<ai::Hint>) -> Result<~[ai::Info], IoError> {
        let r = GetAddrInfoRequest::run(self.uv_loop(), host, servname, hint);
        r.map_err(uv_error_to_io_error)
    }

    fn fs_from_raw_fd(&mut self, fd: c_int,
                      close: CloseBehavior) -> ~RtioFileStream {
        let loop_ = Loop::wrap(self.uv_loop().handle);
        ~FileWatcher::new(loop_, fd, close) as ~RtioFileStream
    }

    fn fs_open(&mut self, path: &CString, fm: FileMode, fa: FileAccess)
        -> Result<~RtioFileStream, IoError> {
        let flags = match fm {
            io::Open => 0,
            io::Append => libc::O_APPEND,
            io::Truncate => libc::O_TRUNC,
        };
        // Opening with a write permission must silently create the file.
        let (flags, mode) = match fa {
            io::Read => (flags | libc::O_RDONLY, 0),
            io::Write => (flags | libc::O_WRONLY | libc::O_CREAT,
                          libc::S_IRUSR | libc::S_IWUSR),
            io::ReadWrite => (flags | libc::O_RDWR | libc::O_CREAT,
                              libc::S_IRUSR | libc::S_IWUSR),
        };

        match FsRequest::open(self.uv_loop(), path, flags as int, mode as int) {
            Ok(fs) => Ok(~fs as ~RtioFileStream),
            Err(e) => Err(uv_error_to_io_error(e))
        }
    }

    fn fs_unlink(&mut self, path: &CString) -> Result<(), IoError> {
        let r = FsRequest::unlink(self.uv_loop(), path);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_lstat(&mut self, path: &CString) -> Result<FileStat, IoError> {
        let r = FsRequest::lstat(self.uv_loop(), path);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_stat(&mut self, path: &CString) -> Result<FileStat, IoError> {
        let r = FsRequest::stat(self.uv_loop(), path);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_mkdir(&mut self, path: &CString,
                perm: io::FilePermission) -> Result<(), IoError> {
        let r = FsRequest::mkdir(self.uv_loop(), path, perm as c_int);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_rmdir(&mut self, path: &CString) -> Result<(), IoError> {
        let r = FsRequest::rmdir(self.uv_loop(), path);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_rename(&mut self, path: &CString, to: &CString) -> Result<(), IoError> {
        let r = FsRequest::rename(self.uv_loop(), path, to);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_chmod(&mut self, path: &CString,
                perm: io::FilePermission) -> Result<(), IoError> {
        let r = FsRequest::chmod(self.uv_loop(), path, perm as c_int);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_readdir(&mut self, path: &CString, flags: c_int)
        -> Result<~[Path], IoError>
    {
        let r = FsRequest::readdir(self.uv_loop(), path, flags);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_link(&mut self, src: &CString, dst: &CString) -> Result<(), IoError> {
        let r = FsRequest::link(self.uv_loop(), src, dst);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_symlink(&mut self, src: &CString, dst: &CString) -> Result<(), IoError> {
        let r = FsRequest::symlink(self.uv_loop(), src, dst);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_chown(&mut self, path: &CString, uid: int, gid: int) -> Result<(), IoError> {
        let r = FsRequest::chown(self.uv_loop(), path, uid, gid);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_readlink(&mut self, path: &CString) -> Result<Path, IoError> {
        let r = FsRequest::readlink(self.uv_loop(), path);
        r.map_err(uv_error_to_io_error)
    }
    fn fs_utime(&mut self, path: &CString, atime: u64, mtime: u64)
        -> Result<(), IoError>
    {
        let r = FsRequest::utime(self.uv_loop(), path, atime, mtime);
        r.map_err(uv_error_to_io_error)
    }

    fn spawn(&mut self, config: ProcessConfig)
            -> Result<(~RtioProcess, ~[Option<~RtioPipe>]), IoError>
    {
        match Process::spawn(self.uv_loop(), config) {
            Ok((p, io)) => {
                Ok((p as ~RtioProcess,
                    io.move_iter().map(|i| i.map(|p| ~p as ~RtioPipe)).collect()))
            }
            Err(e) => Err(uv_error_to_io_error(e)),
        }
    }

    fn unix_bind(&mut self, path: &CString) -> Result<~RtioUnixListener, IoError>
    {
        match PipeListener::bind(self.uv_loop(), path) {
            Ok(p) => Ok(p as ~RtioUnixListener),
            Err(e) => Err(uv_error_to_io_error(e)),
        }
    }

    fn unix_connect(&mut self, path: &CString) -> Result<~RtioPipe, IoError> {
        match PipeWatcher::connect(self.uv_loop(), path) {
            Ok(p) => Ok(~p as ~RtioPipe),
            Err(e) => Err(uv_error_to_io_error(e)),
        }
    }

    fn tty_open(&mut self, fd: c_int, readable: bool)
            -> Result<~RtioTTY, IoError> {
        match TtyWatcher::new(self.uv_loop(), fd, readable) {
            Ok(tty) => Ok(~tty as ~RtioTTY),
            Err(e) => Err(uv_error_to_io_error(e))
        }
    }

    fn pipe_open(&mut self, fd: c_int) -> Result<~RtioPipe, IoError> {
        match PipeWatcher::open(self.uv_loop(), fd) {
            Ok(s) => Ok(~s as ~RtioPipe),
            Err(e) => Err(uv_error_to_io_error(e))
        }
    }

    fn signal(&mut self, signum: Signum, channel: SharedChan<Signum>)
        -> Result<~RtioSignal, IoError> {
        match SignalWatcher::new(self.uv_loop(), signum, channel) {
            Ok(s) => Ok(s as ~RtioSignal),
            Err(e) => Err(uv_error_to_io_error(e)),
        }
    }
}
