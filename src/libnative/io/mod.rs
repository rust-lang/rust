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
use std::comm::SharedChan;
use std::libc::c_int;
use std::libc;
use std::os;
use std::rt::rtio;
use std::rt::rtio::{RtioTcpStream, RtioTcpListener, RtioUdpSocket,
                    RtioUnixListener, RtioPipe, RtioFileStream, RtioProcess,
                    RtioSignal, RtioTTY, CloseBehavior, RtioTimer};
use std::io;
use std::io::IoError;
use std::io::net::ip::SocketAddr;
use std::io::process::ProcessConfig;
use std::io::signal::Signum;
use ai = std::io::net::addrinfo;

// Local re-exports
pub use self::file::FileDesc;
pub use self::process::Process;

// Native I/O implementations
pub mod file;
pub mod process;
pub mod net;

type IoResult<T> = Result<T, IoError>;

fn unimpl() -> IoError {
    IoError {
        kind: io::IoUnavailable,
        desc: "unimplemented I/O interface",
        detail: None,
    }
}

fn translate_error(errno: i32, detail: bool) -> IoError {
    #[cfg(windows)]
    fn get_err(errno: i32) -> (io::IoErrorKind, &'static str) {
        match errno {
            libc::EOF => (io::EndOfFile, "end of file"),
            libc::WSAECONNREFUSED => (io::ConnectionRefused, "connection refused"),
            libc::WSAECONNRESET => (io::ConnectionReset, "connection reset"),
            libc::WSAEACCES => (io::PermissionDenied, "permission denied"),
            libc::WSAEWOULDBLOCK =>
                (io::ResourceUnavailable, "resource temporarily unavailable"),
            libc::WSAENOTCONN => (io::NotConnected, "not connected"),
            libc::WSAECONNABORTED => (io::ConnectionAborted, "connection aborted"),
            libc::WSAEADDRNOTAVAIL => (io::ConnectionRefused, "address not available"),
            libc::WSAEADDRINUSE => (io::ConnectionRefused, "address in use"),

            x => {
                debug!("ignoring {}: {}", x, os::last_os_error());
                (io::OtherIoError, "unknown error")
            }
        }
    }

    #[cfg(not(windows))]
    fn get_err(errno: i32) -> (io::IoErrorKind, &'static str) {
        // XXX: this should probably be a bit more descriptive...
        match errno {
            libc::EOF => (io::EndOfFile, "end of file"),
            libc::ECONNREFUSED => (io::ConnectionRefused, "connection refused"),
            libc::ECONNRESET => (io::ConnectionReset, "connection reset"),
            libc::EPERM | libc::EACCES =>
                (io::PermissionDenied, "permission denied"),
            libc::EPIPE => (io::BrokenPipe, "broken pipe"),
            libc::ENOTCONN => (io::NotConnected, "not connected"),
            libc::ECONNABORTED => (io::ConnectionAborted, "connection aborted"),
            libc::EADDRNOTAVAIL => (io::ConnectionRefused, "address not available"),
            libc::EADDRINUSE => (io::ConnectionRefused, "address in use"),

            // These two constants can have the same value on some systems, but
            // different values on others, so we can't use a match clause
            x if x == libc::EAGAIN || x == libc::EWOULDBLOCK =>
                (io::ResourceUnavailable, "resource temporarily unavailable"),

            x => {
                debug!("ignoring {}: {}", x, os::last_os_error());
                (io::OtherIoError, "unknown error")
            }
        }
    }

    let (kind, desc) = get_err(errno);
    IoError {
        kind: kind,
        desc: desc,
        detail: if detail {Some(os::last_os_error())} else {None},
    }
}

fn last_error() -> IoError { translate_error(os::errno() as i32, true) }

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

#[cfg(unix)]
fn retry(f: || -> libc::c_int) -> IoResult<libc::c_int> {
    loop {
        match f() {
            -1 if os::errno() as int == libc::EINTR as int => {}
            -1 => return Err(last_error()),
            n => return Ok(n),
        }
    }
}

/// Implementation of rt::rtio's IoFactory trait to generate handles to the
/// native I/O functionality.
pub struct IoFactory {
    priv cannot_construct_outside_of_this_module: ()
}

impl IoFactory {
    pub fn new() -> IoFactory {
        net::init();
        IoFactory { cannot_construct_outside_of_this_module: () }
    }
}

impl rtio::IoFactory for IoFactory {
    // networking
    fn tcp_connect(&mut self, addr: SocketAddr) -> IoResult<~RtioTcpStream> {
        net::TcpStream::connect(addr).map(|s| ~s as ~RtioTcpStream)
    }
    fn tcp_bind(&mut self, addr: SocketAddr) -> IoResult<~RtioTcpListener> {
        net::TcpListener::bind(addr).map(|s| ~s as ~RtioTcpListener)
    }
    fn udp_bind(&mut self, addr: SocketAddr) -> IoResult<~RtioUdpSocket> {
        net::UdpSocket::bind(addr).map(|u| ~u as ~RtioUdpSocket)
    }
    fn unix_bind(&mut self, _path: &CString) -> IoResult<~RtioUnixListener> {
        Err(unimpl())
    }
    fn unix_connect(&mut self, _path: &CString) -> IoResult<~RtioPipe> {
        Err(unimpl())
    }
    fn get_host_addresses(&mut self, _host: Option<&str>, _servname: Option<&str>,
                          _hint: Option<ai::Hint>) -> IoResult<~[ai::Info]> {
        Err(unimpl())
    }

    // filesystem operations
    fn fs_from_raw_fd(&mut self, fd: c_int,
                      close: CloseBehavior) -> ~RtioFileStream {
        let close = match close {
            rtio::CloseSynchronously | rtio::CloseAsynchronously => true,
            rtio::DontClose => false
        };
        ~file::FileDesc::new(fd, close) as ~RtioFileStream
    }
    fn fs_open(&mut self, path: &CString, fm: io::FileMode, fa: io::FileAccess)
        -> IoResult<~RtioFileStream> {
        file::open(path, fm, fa).map(|fd| ~fd as ~RtioFileStream)
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
    fn fs_readdir(&mut self, path: &CString, _flags: c_int) -> IoResult<~[Path]> {
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
    fn timer_init(&mut self) -> IoResult<~RtioTimer> {
        Err(unimpl())
    }
    fn spawn(&mut self, config: ProcessConfig)
            -> IoResult<(~RtioProcess, ~[Option<~RtioPipe>])> {
        process::Process::spawn(config).map(|(p, io)| {
            (~p as ~RtioProcess,
             io.move_iter().map(|p| p.map(|p| ~p as ~RtioPipe)).collect())
        })
    }
    fn pipe_open(&mut self, fd: c_int) -> IoResult<~RtioPipe> {
        Ok(~file::FileDesc::new(fd, true) as ~RtioPipe)
    }
    fn tty_open(&mut self, fd: c_int, _readable: bool) -> IoResult<~RtioTTY> {
        if unsafe { libc::isatty(fd) } != 0 {
            Ok(~file::FileDesc::new(fd, true) as ~RtioTTY)
        } else {
            Err(IoError {
                kind: io::MismatchedFileTypeForOperation,
                desc: "file descriptor is not a TTY",
                detail: None,
            })
        }
    }
    fn signal(&mut self, _signal: Signum, _channel: SharedChan<Signum>)
        -> IoResult<~RtioSignal> {
        Err(unimpl())
    }
}
