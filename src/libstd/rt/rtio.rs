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
use option::*;
use result::*;
use comm::{SharedChan, PortOne, Port};
use libc::c_int;
use c_str::CString;

use ai = rt::io::net::addrinfo;
use rt::io::IoError;
use rt::io::signal::Signum;
use super::io::process::ProcessConfig;
use super::io::net::ip::{IpAddr, SocketAddr};
use path::Path;
use super::io::{SeekStyle};
use super::io::{FileMode, FileAccess, FileStat};

pub trait EventLoop {
    fn run(&mut self);
    fn callback(&mut self, ~fn());
    fn pausible_idle_callback(&mut self) -> ~PausibleIdleCallback;
    fn remote_callback(&mut self, ~fn()) -> ~RemoteCallback;

    /// The asynchronous I/O services. Not all event loops may provide one
    // FIXME(#9382) this is an awful interface
    fn io<'a>(&'a mut self, f: &fn(&'a mut IoFactory));
}

pub trait RemoteCallback {
    /// Trigger the remote callback. Note that the number of times the
    /// callback is run is not guaranteed. All that is guaranteed is
    /// that, after calling 'fire', the callback will be called at
    /// least once, but multiple callbacks may be coalesced and
    /// callbacks may be called more often requested. Destruction also
    /// triggers the callback.
    fn fire(&mut self);
}

/// Data needed to make a successful open(2) call
/// Using unix flag conventions for now, which happens to also be what's supported
/// libuv (it does translation to windows under the hood).
pub struct FileOpenConfig {
    /// Path to file to be opened
    path: Path,
    /// Flags for file access mode (as per open(2))
    flags: int,
    /// File creation mode, ignored unless O_CREAT is passed as part of flags
    priv mode: int
}

/// Description of what to do when a file handle is closed
pub enum CloseBehavior {
    /// Do not close this handle when the object is destroyed
    DontClose,
    /// Synchronously close the handle, meaning that the task will block when
    /// the handle is destroyed until it has been fully closed.
    CloseSynchronously,
    /// Asynchronously closes a handle, meaning that the task will *not* block
    /// when the handle is destroyed, but the handle will still get deallocated
    /// and cleaned up (but this will happen asynchronously on the local event
    /// loop).
    CloseAsynchronously,
}

pub fn with_local_io<T>(f: &fn(&mut IoFactory) -> Option<T>) -> Option<T> {
    use rt::sched::Scheduler;
    use rt::local::Local;
    use rt::io::{io_error, standard_error, IoUnavailable};

    unsafe {
        let sched: *mut Scheduler = Local::unsafe_borrow();
        let mut io = None;
        (*sched).event_loop.io(|i| io = Some(i));
        match io {
            Some(io) => f(io),
            None => {
                io_error::cond.raise(standard_error(IoUnavailable));
                None
            }
        }
    }
}

pub trait IoFactory {
    fn tcp_connect(&mut self, addr: SocketAddr) -> Result<~RtioTcpStream, IoError>;
    fn tcp_bind(&mut self, addr: SocketAddr) -> Result<~RtioTcpListener, IoError>;
    fn udp_bind(&mut self, addr: SocketAddr) -> Result<~RtioUdpSocket, IoError>;
    fn get_host_addresses(&mut self, host: Option<&str>, servname: Option<&str>,
                          hint: Option<ai::Hint>) -> Result<~[ai::Info], IoError>;
    fn timer_init(&mut self) -> Result<~RtioTimer, IoError>;
    fn fs_from_raw_fd(&mut self, fd: c_int, close: CloseBehavior) -> ~RtioFileStream;
    fn fs_open(&mut self, path: &CString, fm: FileMode, fa: FileAccess)
        -> Result<~RtioFileStream, IoError>;
    fn fs_unlink(&mut self, path: &CString) -> Result<(), IoError>;
    fn fs_stat(&mut self, path: &CString) -> Result<FileStat, IoError>;
    fn fs_mkdir(&mut self, path: &CString) -> Result<(), IoError>;
    fn fs_rmdir(&mut self, path: &CString) -> Result<(), IoError>;
    fn fs_readdir(&mut self, path: &CString, flags: c_int) ->
        Result<~[Path], IoError>;
    fn spawn(&mut self, config: ProcessConfig)
            -> Result<(~RtioProcess, ~[Option<~RtioPipe>]), IoError>;

    fn pipe_open(&mut self, fd: c_int) -> Result<~RtioPipe, IoError>;
    fn unix_bind(&mut self, path: &CString) ->
        Result<~RtioUnixListener, IoError>;
    fn unix_connect(&mut self, path: &CString) -> Result<~RtioPipe, IoError>;
    fn tty_open(&mut self, fd: c_int, readable: bool)
            -> Result<~RtioTTY, IoError>;
    fn signal(&mut self, signal: Signum, channel: SharedChan<Signum>)
        -> Result<~RtioSignal, IoError>;
}

pub trait RtioTcpListener : RtioSocket {
    fn listen(~self) -> Result<~RtioTcpAcceptor, IoError>;
}

pub trait RtioTcpAcceptor : RtioSocket {
    fn accept(&mut self) -> Result<~RtioTcpStream, IoError>;
    fn accept_simultaneously(&mut self) -> Result<(), IoError>;
    fn dont_accept_simultaneously(&mut self) -> Result<(), IoError>;
}

pub trait RtioTcpStream : RtioSocket {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError>;
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError>;
    fn peer_name(&mut self) -> Result<SocketAddr, IoError>;
    fn control_congestion(&mut self) -> Result<(), IoError>;
    fn nodelay(&mut self) -> Result<(), IoError>;
    fn keepalive(&mut self, delay_in_seconds: uint) -> Result<(), IoError>;
    fn letdie(&mut self) -> Result<(), IoError>;
}

pub trait RtioSocket {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError>;
}

pub trait RtioUdpSocket : RtioSocket {
    fn recvfrom(&mut self, buf: &mut [u8]) -> Result<(uint, SocketAddr), IoError>;
    fn sendto(&mut self, buf: &[u8], dst: SocketAddr) -> Result<(), IoError>;

    fn join_multicast(&mut self, multi: IpAddr) -> Result<(), IoError>;
    fn leave_multicast(&mut self, multi: IpAddr) -> Result<(), IoError>;

    fn loop_multicast_locally(&mut self) -> Result<(), IoError>;
    fn dont_loop_multicast_locally(&mut self) -> Result<(), IoError>;

    fn multicast_time_to_live(&mut self, ttl: int) -> Result<(), IoError>;
    fn time_to_live(&mut self, ttl: int) -> Result<(), IoError>;

    fn hear_broadcasts(&mut self) -> Result<(), IoError>;
    fn ignore_broadcasts(&mut self) -> Result<(), IoError>;
}

pub trait RtioTimer {
    fn sleep(&mut self, msecs: u64);
    fn oneshot(&mut self, msecs: u64) -> PortOne<()>;
    fn period(&mut self, msecs: u64) -> Port<()>;
}

pub trait RtioFileStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<int, IoError>;
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError>;
    fn pread(&mut self, buf: &mut [u8], offset: u64) -> Result<int, IoError>;
    fn pwrite(&mut self, buf: &[u8], offset: u64) -> Result<(), IoError>;
    fn seek(&mut self, pos: i64, whence: SeekStyle) -> Result<u64, IoError>;
    fn tell(&self) -> Result<u64, IoError>;
}

pub trait RtioProcess {
    fn id(&self) -> libc::pid_t;
    fn kill(&mut self, signal: int) -> Result<(), IoError>;
    fn wait(&mut self) -> int;
}

pub trait RtioPipe {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError>;
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError>;
}

pub trait RtioUnixListener {
    fn listen(~self) -> Result<~RtioUnixAcceptor, IoError>;
}

pub trait RtioUnixAcceptor {
    fn accept(&mut self) -> Result<~RtioPipe, IoError>;
    fn accept_simultaneously(&mut self) -> Result<(), IoError>;
    fn dont_accept_simultaneously(&mut self) -> Result<(), IoError>;
}

pub trait RtioTTY {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError>;
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError>;
    fn set_raw(&mut self, raw: bool) -> Result<(), IoError>;
    fn get_winsize(&mut self) -> Result<(int, int), IoError>;
    fn isatty(&self) -> bool;
}

pub trait PausibleIdleCallback {
    fn start(&mut self, f: ~fn());
    fn pause(&mut self);
    fn resume(&mut self);
    fn close(&mut self);
}

pub trait RtioSignal {}
