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
use std::cast::transmute;
use std::cast;
use std::comm::{SharedChan, GenericChan};
use std::libc::c_int;
use std::libc;
use std::path::Path;
use std::rt::io::IoError;
use std::rt::io::net::ip::SocketAddr;
use std::rt::io::process::ProcessConfig;
use std::rt::io;
use std::rt::local::Local;
use std::rt::rtio::*;
use std::rt::sched::{Scheduler, SchedHandle};
use std::rt::task::Task;
use std::str;
use std::libc::{O_CREAT, O_APPEND, O_TRUNC, O_RDWR, O_RDONLY, O_WRONLY,
                S_IRUSR, S_IWUSR};
use std::rt::io::{FileMode, FileAccess, Open, Append, Truncate, Read, Write,
                  ReadWrite, FileStat};
use std::rt::io::signal::Signum;
use std::task;
use ai = std::rt::io::net::addrinfo;

#[cfg(test)] use std::unstable::run_in_bare_thread;
#[cfg(test)] use std::rt::test::{spawntask,
                                 next_test_ip4,
                                 run_in_mt_newsched_task};
#[cfg(test)] use std::rt::comm::oneshot;

use super::*;
use addrinfo::GetAddrInfoRequest;

pub trait HomingIO {

    fn home<'r>(&'r mut self) -> &'r mut SchedHandle;

    /// This function will move tasks to run on their home I/O scheduler. Note
    /// that this function does *not* pin the task to the I/O scheduler, but
    /// rather it simply moves it to running on the I/O scheduler.
    fn go_to_IO_home(&mut self) -> uint {
        use std::rt::sched::RunOnce;

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

impl Drop for HomingMissile {
    fn drop(&mut self) {
        // It would truly be a sad day if we had moved off the home I/O
        // scheduler while we were doing I/O.
        assert_eq!(Local::borrow(|sched: &mut Scheduler| sched.sched_id()),
                   self.io_home);

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

    fn pausible_idle_callback(&mut self) -> ~PausibleIdleCallback {
        IdleWatcher::new(self.uvio.uv_loop()) as ~PausibleIdleCallback
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
    use std::rt::sleeper_list::SleeperList;
    use std::rt::work_queue::WorkQueue;
    use std::rt::thread::Thread;
    use std::rt::task::Task;
    use std::rt::sched::{Shutdown, TaskFromFriend};
    use std::rt::task::UnwindResult;
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

        let on_exit: ~fn(UnwindResult) = |exit_status| {
            handle1.take().send(Shutdown);
            handle2.take().send(Shutdown);
            assert!(exit_status.is_success());
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
    use std::rt::sleeper_list::SleeperList;
    use std::rt::work_queue::WorkQueue;
    use std::rt::thread::Thread;
    use std::rt::task::Task;
    use std::rt::comm::oneshot;
    use std::rt::sched::Shutdown;
    use std::rt::task::UnwindResult;
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

        let on_exit: ~fn(UnwindResult) = |exit| {
            handle1.take().send(Shutdown);
            handle2.take().send(Shutdown);
            assert!(exit.is_success());
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
                    uvdebug!("{}", buf[i]);
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
    use std::rt::sleeper_list::SleeperList;
    use std::rt::work_queue::WorkQueue;
    use std::rt::thread::Thread;
    use std::rt::task::Task;
    use std::rt::sched::{Shutdown};
    use std::rt::task::UnwindResult;
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

        let server_on_exit: ~fn(UnwindResult) = |exit_status| {
            server_handle.take().send(Shutdown);
            assert!(exit_status.is_success());
        };

        let client_on_exit: ~fn(UnwindResult) = |exit_status| {
            client_handle.take().send(Shutdown);
            assert!(exit_status.is_success());
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
                    uvdebug!("{}", buf[i]);
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
                    uvdebug!("read {} bytes", nread);
                    total_bytes_read += nread;
                    for i in range(0u, nread) {
                        assert_eq!(buf[i], 1);
                    }
                }
                uvdebug!("read {} bytes total", total_bytes_read);
            }
        }
    }
}

#[test]
#[ignore(cfg(windows))] // FIXME(#10102) the server never sees the second send
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
            do timer.map |mut t| { t.sleep(1) };
        }
    }
}

fn file_test_uvio_full_simple_impl() {
    use std::rt::io::{Open, ReadWrite, Read};
    unsafe {
        let io = local_io();
        let write_val = "hello uvio!";
        let path = "./tmp/file_test_uvio_full.txt";
        {
            let create_fm = Open;
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
    unsafe {
        use std::libc::{STDOUT_FILENO};
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
