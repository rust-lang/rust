// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The EventLoop and internal synchronous I/O interface.

use core::prelude::*;
use alloc::boxed::Box;
use collections::string::String;
use collections::vec::Vec;
use core::fmt;
use core::mem;
use libc::c_int;
use libc;

use c_str::CString;
use local::Local;
use task::Task;

pub trait EventLoop {
    fn run(&mut self);
    fn callback(&mut self, arg: proc(): Send);
    fn pausable_idle_callback(&mut self, Box<Callback + Send>)
                              -> Box<PausableIdleCallback + Send>;
    fn remote_callback(&mut self, Box<Callback + Send>)
                       -> Box<RemoteCallback + Send>;

    /// The asynchronous I/O services. Not all event loops may provide one.
    fn io<'a>(&'a mut self) -> Option<&'a mut IoFactory>;
    fn has_active_io(&self) -> bool;
}

pub trait Callback {
    fn call(&mut self);
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

/// Data needed to spawn a process. Serializes the `std::io::process::Command`
/// builder.
pub struct ProcessConfig<'a> {
    /// Path to the program to run.
    pub program: &'a CString,

    /// Arguments to pass to the program (doesn't include the program itself).
    pub args: &'a [CString],

    /// Optional environment to specify for the program. If this is None, then
    /// it will inherit the current process's environment.
    pub env: Option<&'a [(&'a CString, &'a CString)]>,

    /// Optional working directory for the new process. If this is None, then
    /// the current directory of the running process is inherited.
    pub cwd: Option<&'a CString>,

    /// Configuration for the child process's stdin handle (file descriptor 0).
    /// This field defaults to `CreatePipe(true, false)` so the input can be
    /// written to.
    pub stdin: StdioContainer,

    /// Configuration for the child process's stdout handle (file descriptor 1).
    /// This field defaults to `CreatePipe(false, true)` so the output can be
    /// collected.
    pub stdout: StdioContainer,

    /// Configuration for the child process's stdout handle (file descriptor 2).
    /// This field defaults to `CreatePipe(false, true)` so the output can be
    /// collected.
    pub stderr: StdioContainer,

    /// Any number of streams/file descriptors/pipes may be attached to this
    /// process. This list enumerates the file descriptors and such for the
    /// process to be spawned, and the file descriptors inherited will start at
    /// 3 and go to the length of this array. The first three file descriptors
    /// (stdin/stdout/stderr) are configured with the `stdin`, `stdout`, and
    /// `stderr` fields.
    pub extra_io: &'a [StdioContainer],

    /// Sets the child process's user id. This translates to a `setuid` call in
    /// the child process. Setting this value on windows will cause the spawn to
    /// fail. Failure in the `setuid` call on unix will also cause the spawn to
    /// fail.
    pub uid: Option<uint>,

    /// Similar to `uid`, but sets the group id of the child process. This has
    /// the same semantics as the `uid` field.
    pub gid: Option<uint>,

    /// If true, the child process is spawned in a detached state. On unix, this
    /// means that the child is the leader of a new process group.
    pub detach: bool,
}

pub struct LocalIo<'a> {
    factory: &'a mut IoFactory+'a,
}

#[unsafe_destructor]
impl<'a> Drop for LocalIo<'a> {
    fn drop(&mut self) {
        // FIXME(pcwalton): Do nothing here for now, but eventually we may want
        // something. For now this serves to make `LocalIo` noncopyable.
    }
}

impl<'a> LocalIo<'a> {
    /// Returns the local I/O: either the local scheduler's I/O services or
    /// the native I/O services.
    pub fn borrow() -> Option<LocalIo<'a>> {
        // FIXME(#11053): bad
        //
        // This is currently very unsafely implemented. We don't actually
        // *take* the local I/O so there's a very real possibility that we
        // can have two borrows at once. Currently there is not a clear way
        // to actually borrow the local I/O factory safely because even if
        // ownership were transferred down to the functions that the I/O
        // factory implements it's just too much of a pain to know when to
        // relinquish ownership back into the local task (but that would be
        // the safe way of implementing this function).
        //
        // In order to get around this, we just transmute a copy out of the task
        // in order to have what is likely a static lifetime (bad).
        let mut t: Box<Task> = match Local::try_take() {
            Some(t) => t,
            None => return None,
        };
        let ret = t.local_io().map(|t| {
            unsafe { mem::transmute_copy(&t) }
        });
        Local::put(t);
        return ret;
    }

    pub fn maybe_raise<T>(f: |io: &mut IoFactory| -> IoResult<T>)
        -> IoResult<T>
    {
        #[cfg(unix)] use libc::EINVAL as ERROR;
        #[cfg(windows)] use libc::ERROR_CALL_NOT_IMPLEMENTED as ERROR;
        match LocalIo::borrow() {
            Some(mut io) => f(io.get()),
            None => Err(IoError {
                code: ERROR as uint,
                extra: 0,
                detail: None,
            }),
        }
    }

    pub fn new<'a>(io: &'a mut IoFactory+'a) -> LocalIo<'a> {
        LocalIo { factory: io }
    }

    /// Returns the underlying I/O factory as a trait reference.
    #[inline]
    pub fn get<'a>(&'a mut self) -> &'a mut IoFactory {
        let f: &'a mut IoFactory = self.factory;
        f
    }
}

pub trait IoFactory {
    // networking
    fn tcp_connect(&mut self, addr: SocketAddr,
                   timeout: Option<u64>) -> IoResult<Box<RtioTcpStream + Send>>;
    fn tcp_bind(&mut self, addr: SocketAddr)
                -> IoResult<Box<RtioTcpListener + Send>>;
    fn udp_bind(&mut self, addr: SocketAddr)
                -> IoResult<Box<RtioUdpSocket + Send>>;
    fn unix_bind(&mut self, path: &CString)
                 -> IoResult<Box<RtioUnixListener + Send>>;
    fn unix_connect(&mut self, path: &CString,
                    timeout: Option<u64>) -> IoResult<Box<RtioPipe + Send>>;
    fn get_host_addresses(&mut self, host: Option<&str>, servname: Option<&str>,
                          hint: Option<AddrinfoHint>)
                          -> IoResult<Vec<AddrinfoInfo>>;

    // filesystem operations
    fn fs_from_raw_fd(&mut self, fd: c_int, close: CloseBehavior)
                      -> Box<RtioFileStream + Send>;
    fn fs_open(&mut self, path: &CString, fm: FileMode, fa: FileAccess)
               -> IoResult<Box<RtioFileStream + Send>>;
    fn fs_unlink(&mut self, path: &CString) -> IoResult<()>;
    fn fs_stat(&mut self, path: &CString) -> IoResult<FileStat>;
    fn fs_mkdir(&mut self, path: &CString, mode: uint) -> IoResult<()>;
    fn fs_chmod(&mut self, path: &CString, mode: uint) -> IoResult<()>;
    fn fs_rmdir(&mut self, path: &CString) -> IoResult<()>;
    fn fs_rename(&mut self, path: &CString, to: &CString) -> IoResult<()>;
    fn fs_readdir(&mut self, path: &CString, flags: c_int) ->
        IoResult<Vec<CString>>;
    fn fs_lstat(&mut self, path: &CString) -> IoResult<FileStat>;
    fn fs_chown(&mut self, path: &CString, uid: int, gid: int) ->
        IoResult<()>;
    fn fs_readlink(&mut self, path: &CString) -> IoResult<CString>;
    fn fs_symlink(&mut self, src: &CString, dst: &CString) -> IoResult<()>;
    fn fs_link(&mut self, src: &CString, dst: &CString) -> IoResult<()>;
    fn fs_utime(&mut self, src: &CString, atime: u64, mtime: u64) ->
        IoResult<()>;

    // misc
    fn timer_init(&mut self) -> IoResult<Box<RtioTimer + Send>>;
    fn spawn(&mut self, cfg: ProcessConfig)
            -> IoResult<(Box<RtioProcess + Send>,
                         Vec<Option<Box<RtioPipe + Send>>>)>;
    fn kill(&mut self, pid: libc::pid_t, signal: int) -> IoResult<()>;
    fn pipe_open(&mut self, fd: c_int) -> IoResult<Box<RtioPipe + Send>>;
    fn tty_open(&mut self, fd: c_int, readable: bool)
            -> IoResult<Box<RtioTTY + Send>>;
    fn signal(&mut self, signal: int, cb: Box<Callback + Send>)
        -> IoResult<Box<RtioSignal + Send>>;
}

pub trait RtioTcpListener : RtioSocket {
    fn listen(self: Box<Self>) -> IoResult<Box<RtioTcpAcceptor + Send>>;
}

pub trait RtioTcpAcceptor : RtioSocket {
    fn accept(&mut self) -> IoResult<Box<RtioTcpStream + Send>>;
    fn accept_simultaneously(&mut self) -> IoResult<()>;
    fn dont_accept_simultaneously(&mut self) -> IoResult<()>;
    fn set_timeout(&mut self, timeout: Option<u64>);
    fn clone(&self) -> Box<RtioTcpAcceptor + Send>;
    fn close_accept(&mut self) -> IoResult<()>;
}

pub trait RtioTcpStream : RtioSocket {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint>;
    fn write(&mut self, buf: &[u8]) -> IoResult<()>;
    fn peer_name(&mut self) -> IoResult<SocketAddr>;
    fn control_congestion(&mut self) -> IoResult<()>;
    fn nodelay(&mut self) -> IoResult<()>;
    fn keepalive(&mut self, delay_in_seconds: uint) -> IoResult<()>;
    fn letdie(&mut self) -> IoResult<()>;
    fn clone(&self) -> Box<RtioTcpStream + Send>;
    fn close_write(&mut self) -> IoResult<()>;
    fn close_read(&mut self) -> IoResult<()>;
    fn set_timeout(&mut self, timeout_ms: Option<u64>);
    fn set_read_timeout(&mut self, timeout_ms: Option<u64>);
    fn set_write_timeout(&mut self, timeout_ms: Option<u64>);
}

pub trait RtioSocket {
    fn socket_name(&mut self) -> IoResult<SocketAddr>;
}

pub trait RtioUdpSocket : RtioSocket {
    fn recv_from(&mut self, buf: &mut [u8]) -> IoResult<(uint, SocketAddr)>;
    fn send_to(&mut self, buf: &[u8], dst: SocketAddr) -> IoResult<()>;

    fn join_multicast(&mut self, multi: IpAddr) -> IoResult<()>;
    fn leave_multicast(&mut self, multi: IpAddr) -> IoResult<()>;

    fn loop_multicast_locally(&mut self) -> IoResult<()>;
    fn dont_loop_multicast_locally(&mut self) -> IoResult<()>;

    fn multicast_time_to_live(&mut self, ttl: int) -> IoResult<()>;
    fn time_to_live(&mut self, ttl: int) -> IoResult<()>;

    fn hear_broadcasts(&mut self) -> IoResult<()>;
    fn ignore_broadcasts(&mut self) -> IoResult<()>;

    fn clone(&self) -> Box<RtioUdpSocket + Send>;
    fn set_timeout(&mut self, timeout_ms: Option<u64>);
    fn set_read_timeout(&mut self, timeout_ms: Option<u64>);
    fn set_write_timeout(&mut self, timeout_ms: Option<u64>);
}

pub trait RtioTimer {
    fn sleep(&mut self, msecs: u64);
    fn oneshot(&mut self, msecs: u64, cb: Box<Callback + Send>);
    fn period(&mut self, msecs: u64, cb: Box<Callback + Send>);
}

pub trait RtioFileStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<int>;
    fn write(&mut self, buf: &[u8]) -> IoResult<()>;
    fn pread(&mut self, buf: &mut [u8], offset: u64) -> IoResult<int>;
    fn pwrite(&mut self, buf: &[u8], offset: u64) -> IoResult<()>;
    fn seek(&mut self, pos: i64, whence: SeekStyle) -> IoResult<u64>;
    fn tell(&self) -> IoResult<u64>;
    fn fsync(&mut self) -> IoResult<()>;
    fn datasync(&mut self) -> IoResult<()>;
    fn truncate(&mut self, offset: i64) -> IoResult<()>;
    fn fstat(&mut self) -> IoResult<FileStat>;
}

pub trait RtioProcess {
    fn id(&self) -> libc::pid_t;
    fn kill(&mut self, signal: int) -> IoResult<()>;
    fn wait(&mut self) -> IoResult<ProcessExit>;
    fn set_timeout(&mut self, timeout: Option<u64>);
}

pub trait RtioPipe {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint>;
    fn write(&mut self, buf: &[u8]) -> IoResult<()>;
    fn clone(&self) -> Box<RtioPipe + Send>;

    fn close_write(&mut self) -> IoResult<()>;
    fn close_read(&mut self) -> IoResult<()>;
    fn set_timeout(&mut self, timeout_ms: Option<u64>);
    fn set_read_timeout(&mut self, timeout_ms: Option<u64>);
    fn set_write_timeout(&mut self, timeout_ms: Option<u64>);
}

pub trait RtioUnixListener {
    fn listen(self: Box<Self>) -> IoResult<Box<RtioUnixAcceptor + Send>>;
}

pub trait RtioUnixAcceptor {
    fn accept(&mut self) -> IoResult<Box<RtioPipe + Send>>;
    fn set_timeout(&mut self, timeout: Option<u64>);
    fn clone(&self) -> Box<RtioUnixAcceptor + Send>;
    fn close_accept(&mut self) -> IoResult<()>;
}

pub trait RtioTTY {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint>;
    fn write(&mut self, buf: &[u8]) -> IoResult<()>;
    fn set_raw(&mut self, raw: bool) -> IoResult<()>;
    fn get_winsize(&mut self) -> IoResult<(int, int)>;
    fn isatty(&self) -> bool;
}

pub trait PausableIdleCallback {
    fn pause(&mut self);
    fn resume(&mut self);
}

pub trait RtioSignal {}

pub struct IoError {
    pub code: uint,
    pub extra: uint,
    pub detail: Option<String>,
}

pub type IoResult<T> = Result<T, IoError>;

#[deriving(PartialEq, Eq)]
pub enum IpAddr {
    Ipv4Addr(u8, u8, u8, u8),
    Ipv6Addr(u16, u16, u16, u16, u16, u16, u16, u16),
}

impl fmt::Show for IpAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Ipv4Addr(a, b, c, d) => write!(fmt, "{}.{}.{}.{}", a, b, c, d),
            Ipv6Addr(a, b, c, d, e, f, g, h) => {
                write!(fmt,
                       "{:04x}:{:04x}:{:04x}:{:04x}:{:04x}:{:04x}:{:04x}:{:04x}",
                       a, b, c, d, e, f, g, h)
            }
        }
    }
}

#[deriving(PartialEq, Eq)]
pub struct SocketAddr {
    pub ip: IpAddr,
    pub port: u16,
}

pub enum StdioContainer {
    Ignored,
    InheritFd(i32),
    CreatePipe(bool, bool),
}

pub enum ProcessExit {
    ExitStatus(int),
    ExitSignal(int),
}

pub enum FileMode {
    Open,
    Append,
    Truncate,
}

pub enum FileAccess {
    Read,
    Write,
    ReadWrite,
}

pub struct FileStat {
    pub size: u64,
    pub kind: u64,
    pub perm: u64,
    pub created: u64,
    pub modified: u64,
    pub accessed: u64,
    pub device: u64,
    pub inode: u64,
    pub rdev: u64,
    pub nlink: u64,
    pub uid: u64,
    pub gid: u64,
    pub blksize: u64,
    pub blocks: u64,
    pub flags: u64,
    pub gen: u64,
}

pub enum SeekStyle {
    SeekSet,
    SeekEnd,
    SeekCur,
}

pub struct AddrinfoHint {
    pub family: uint,
    pub socktype: uint,
    pub protocol: uint,
    pub flags: uint,
}

pub struct AddrinfoInfo {
    pub address: SocketAddr,
    pub family: uint,
    pub socktype: uint,
    pub protocol: uint,
    pub flags: uint,
}
