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

use c_str::CString;
use comm::SharedChan;
use libc::c_int;
use libc;
use option::{Option, None, Some};
use os;
use path::Path;
use result::{Result, Ok, Err};
use rt::rtio;
use rt::rtio::{RtioTcpStream, RtioTcpListener, RtioUdpSocket, RtioUnixListener,
               RtioPipe, RtioFileStream, RtioProcess, RtioSignal, RtioTTY,
               CloseBehavior, RtioTimer};
use io;
use io::IoError;
use io::net::ip::SocketAddr;
use io::process::ProcessConfig;
use io::signal::Signum;
use ai = io::net::addrinfo;

// Local re-exports
pub use self::file::FileDesc;
pub use self::process::Process;

// Native I/O implementations
pub mod file;
pub mod process;

type IoResult<T> = Result<T, IoError>;

fn unimpl() -> IoError {
    IoError {
        kind: io::IoUnavailable,
        desc: "unimplemented I/O interface",
        detail: None,
    }
}

fn last_error() -> IoError {
    #[cfg(windows)]
    fn get_err(errno: i32) -> (io::IoErrorKind, &'static str) {
        match errno {
            libc::EOF => (io::EndOfFile, "end of file"),
            _ => (io::OtherIoError, "unknown error"),
        }
    }

    #[cfg(not(windows))]
    fn get_err(errno: i32) -> (io::IoErrorKind, &'static str) {
        // XXX: this should probably be a bit more descriptive...
        match errno {
            libc::EOF => (io::EndOfFile, "end of file"),

            // These two constants can have the same value on some systems, but
            // different values on others, so we can't use a match clause
            x if x == libc::EAGAIN || x == libc::EWOULDBLOCK =>
                (io::ResourceUnavailable, "resource temporarily unavailable"),

            _ => (io::OtherIoError, "unknown error"),
        }
    }

    let (kind, desc) = get_err(os::errno() as i32);
    IoError {
        kind: kind,
        desc: desc,
        detail: Some(os::last_os_error())
    }
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

/// Implementation of rt::rtio's IoFactory trait to generate handles to the
/// native I/O functionality.
pub struct IoFactory;

impl rtio::IoFactory for IoFactory {
    // networking
    fn tcp_connect(&mut self, _addr: SocketAddr) -> IoResult<~RtioTcpStream> {
        Err(unimpl())
    }
    fn tcp_bind(&mut self, _addr: SocketAddr) -> IoResult<~RtioTcpListener> {
        Err(unimpl())
    }
    fn udp_bind(&mut self, _addr: SocketAddr) -> IoResult<~RtioUdpSocket> {
        Err(unimpl())
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
            // Don't ever close the stdio file descriptors, nothing good really
            // comes of that.
            Ok(~file::FileDesc::new(fd, fd > libc::STDERR_FILENO) as ~RtioTTY)
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
