// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Native thread-blocking I/O implementation
//!
//! This module contains the implementation of native thread-blocking
//! implementations of I/O on all platforms. This module is not intended to be
//! used directly, but rather the rust runtime will fall back to using it if
//! necessary.
//!
//! Rust code normally runs inside of green tasks with a local scheduler using
//! asynchronous I/O to cooperate among tasks. This model is not always
//! available, however, and that's where these native implementations come into
//! play. The only dependencies of these modules are the normal system libraries
//! that you would find on the respective platform.

use std::c_str::CString;
use std::io;
use std::io::IoError;
use std::io::net::ip::SocketAddr;
use std::io::process::ProcessConfig;
use std::io::signal::Signum;
use libc::c_int;
use libc;
use std::os;
use std::rt::rtio;
use std::rt::rtio::{RtioTcpStream, RtioTcpListener, RtioUdpSocket,
                    RtioUnixListener, RtioPipe, RtioFileStream, RtioProcess,
                    RtioSignal, RtioTTY, CloseBehavior, RtioTimer};
use ai = std::io::net::addrinfo;

// Local re-exports
pub use self::file::FileDesc;
pub use self::process::Process;

// Native I/O implementations
pub mod addrinfo;
pub mod net;
pub mod process;

#[cfg(unix)]
#[path = "file_unix.rs"]
pub mod file;
#[cfg(windows)]
#[path = "file_win32.rs"]
pub mod file;

#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "android")]
#[cfg(target_os = "linux")]
#[path = "timer_unix.rs"]
pub mod timer;

#[cfg(target_os = "win32")]
#[path = "timer_win32.rs"]
pub mod timer;

#[cfg(unix)]
#[path = "pipe_unix.rs"]
pub mod pipe;

#[cfg(windows)]
#[path = "pipe_win32.rs"]
pub mod pipe;

#[cfg(unix)]    #[path = "c_unix.rs"]  mod c;
#[cfg(windows)] #[path = "c_win32.rs"] mod c;

mod timer_helper;

pub type IoResult<T> = Result<T, IoError>;

fn unimpl() -> IoError {
    IoError {
        kind: io::IoUnavailable,
        desc: "unimplemented I/O interface",
        detail: None,
    }
}

fn last_error() -> IoError {
    IoError::last_error()
}

// unix has nonzero values as errors
fn mkerr_libc(ret: libc::c_int) -> IoResult<()> {
    if ret != 0 {
        Err(last_error())
    } else {
        Ok(())
    }
}

// windows has zero values as errors
#[cfg(windows)]
fn mkerr_winbool(ret: libc::c_int) -> IoResult<()> {
    if ret == 0 {
        Err(last_error())
    } else {
        Ok(())
    }
}

#[cfg(windows)]
#[inline]
fn retry(f: || -> libc::c_int) -> libc::c_int {
    loop {
        match f() {
            -1 if os::errno() as int == libc::WSAEINTR as int => {}
            n => return n,
        }
    }
}

#[cfg(unix)]
#[inline]
fn retry(f: || -> libc::c_int) -> libc::c_int {
    loop {
        match f() {
            -1 if os::errno() as int == libc::EINTR as int => {}
            n => return n,
        }
    }
}

fn keep_going(data: &[u8], f: |*u8, uint| -> i64) -> i64 {
    let origamt = data.len();
    let mut data = data.as_ptr();
    let mut amt = origamt;
    while amt > 0 {
        let ret = retry(|| f(data, amt) as libc::c_int);
        if ret == 0 {
            break
        } else if ret != -1 {
            amt -= ret as uint;
            data = unsafe { data.offset(ret as int) };
        } else {
            return ret as i64;
        }
    }
    return (origamt - amt) as i64;
}

/// Implementation of rt::rtio's IoFactory trait to generate handles to the
/// native I/O functionality.
pub struct IoFactory {
    cannot_construct_outside_of_this_module: ()
}

impl IoFactory {
    pub fn new() -> IoFactory {
        net::init();
        IoFactory { cannot_construct_outside_of_this_module: () }
    }
}

impl rtio::IoFactory for IoFactory {
    // networking
    fn tcp_connect(&mut self, addr: SocketAddr,
                   timeout: Option<u64>) -> IoResult<~RtioTcpStream:Send> {
        net::TcpStream::connect(addr, timeout).map(|s| ~s as ~RtioTcpStream:Send)
    }
    fn tcp_bind(&mut self, addr: SocketAddr) -> IoResult<~RtioTcpListener:Send> {
        net::TcpListener::bind(addr).map(|s| ~s as ~RtioTcpListener:Send)
    }
    fn udp_bind(&mut self, addr: SocketAddr) -> IoResult<~RtioUdpSocket:Send> {
        net::UdpSocket::bind(addr).map(|u| ~u as ~RtioUdpSocket:Send)
    }
    fn unix_bind(&mut self, path: &CString) -> IoResult<~RtioUnixListener:Send> {
        pipe::UnixListener::bind(path).map(|s| ~s as ~RtioUnixListener:Send)
    }
    fn unix_connect(&mut self, path: &CString) -> IoResult<~RtioPipe:Send> {
        pipe::UnixStream::connect(path).map(|s| ~s as ~RtioPipe:Send)
    }
    fn get_host_addresses(&mut self, host: Option<&str>, servname: Option<&str>,
                          hint: Option<ai::Hint>) -> IoResult<~[ai::Info]> {
        addrinfo::GetAddrInfoRequest::run(host, servname, hint)
    }

    // filesystem operations
    fn fs_from_raw_fd(&mut self, fd: c_int,
                      close: CloseBehavior) -> ~RtioFileStream:Send {
        let close = match close {
            rtio::CloseSynchronously | rtio::CloseAsynchronously => true,
            rtio::DontClose => false
        };
        ~file::FileDesc::new(fd, close) as ~RtioFileStream:Send
    }
    fn fs_open(&mut self, path: &CString, fm: io::FileMode, fa: io::FileAccess)
        -> IoResult<~RtioFileStream:Send> {
        file::open(path, fm, fa).map(|fd| ~fd as ~RtioFileStream:Send)
    }
    fn fs_unlink(&mut self, path: &CString) -> IoResult<()> {
        file::unlink(path)
    }
    fn fs_stat(&mut self, path: &CString) -> IoResult<io::FileStat> {
        file::stat(path)
    }
    fn fs_mkdir(&mut self, path: &CString,
                mode: io::FilePermission) -> IoResult<()> {
        file::mkdir(path, mode)
    }
    fn fs_chmod(&mut self, path: &CString,
                mode: io::FilePermission) -> IoResult<()> {
        file::chmod(path, mode)
    }
    fn fs_rmdir(&mut self, path: &CString) -> IoResult<()> {
        file::rmdir(path)
    }
    fn fs_rename(&mut self, path: &CString, to: &CString) -> IoResult<()> {
        file::rename(path, to)
    }
    fn fs_readdir(&mut self, path: &CString, _flags: c_int) -> IoResult<Vec<Path>> {
        file::readdir(path)
    }
    fn fs_lstat(&mut self, path: &CString) -> IoResult<io::FileStat> {
        file::lstat(path)
    }
    fn fs_chown(&mut self, path: &CString, uid: int, gid: int) -> IoResult<()> {
        file::chown(path, uid, gid)
    }
    fn fs_readlink(&mut self, path: &CString) -> IoResult<Path> {
        file::readlink(path)
    }
    fn fs_symlink(&mut self, src: &CString, dst: &CString) -> IoResult<()> {
        file::symlink(src, dst)
    }
    fn fs_link(&mut self, src: &CString, dst: &CString) -> IoResult<()> {
        file::link(src, dst)
    }
    fn fs_utime(&mut self, src: &CString, atime: u64,
                mtime: u64) -> IoResult<()> {
        file::utime(src, atime, mtime)
    }

    // misc
    fn timer_init(&mut self) -> IoResult<~RtioTimer:Send> {
        timer::Timer::new().map(|t| ~t as ~RtioTimer:Send)
    }
    fn spawn(&mut self, config: ProcessConfig)
            -> IoResult<(~RtioProcess:Send, ~[Option<~RtioPipe:Send>])> {
        process::Process::spawn(config).map(|(p, io)| {
            (~p as ~RtioProcess:Send,
             io.move_iter().map(|p| p.map(|p| ~p as ~RtioPipe:Send)).collect())
        })
    }
    fn kill(&mut self, pid: libc::pid_t, signum: int) -> IoResult<()> {
        process::Process::kill(pid, signum)
    }
    fn pipe_open(&mut self, fd: c_int) -> IoResult<~RtioPipe:Send> {
        Ok(~file::FileDesc::new(fd, true) as ~RtioPipe:Send)
    }
    fn tty_open(&mut self, fd: c_int, _readable: bool)
        -> IoResult<~RtioTTY:Send>
    {
        if unsafe { libc::isatty(fd) } != 0 {
            Ok(~file::FileDesc::new(fd, true) as ~RtioTTY:Send)
        } else {
            Err(IoError {
                kind: io::MismatchedFileTypeForOperation,
                desc: "file descriptor is not a TTY",
                detail: None,
            })
        }
    }
    fn signal(&mut self, _signal: Signum, _channel: Sender<Signum>)
        -> IoResult<~RtioSignal:Send> {
        Err(unimpl())
    }
}
