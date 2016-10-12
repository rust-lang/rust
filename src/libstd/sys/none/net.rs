use io;

use libc::{c_int, sockaddr, socklen_t};
use net::Shutdown;
use time::Duration;
use sys::fd::FileDesc;
use sys_common::{AsInner, FromInner, IntoInner};

pub mod netc {
    pub use os::none::libc::*;
}

pub struct Socket(FileDesc);

pub fn generic_error() -> io::Error {
    io::Error::new(io::ErrorKind::Other, "networking not supported on this platform")
}

impl Socket {
    pub fn new_raw(_fam: c_int, _ty: c_int) -> io::Result<Socket> { Err(generic_error()) }
    pub fn new_pair(_fam: c_int, _ty: c_int) -> io::Result<(Socket, Socket)> { Err(generic_error()) }
    pub fn accept(&self, _storage: *mut sockaddr, _len: *mut socklen_t)
                  -> io::Result<Socket> { Err(generic_error()) }
    pub fn duplicate(&self) -> io::Result<Socket> { Err(generic_error()) }
    pub fn read(&self, _buf: &mut [u8]) -> io::Result<usize> { Err(generic_error()) }
    pub fn read_to_end(&self, _buf: &mut Vec<u8>) -> io::Result<usize> { Err(generic_error()) }
    pub fn write(&self, _buf: &[u8]) -> io::Result<usize> { Err(generic_error()) }
    pub fn set_timeout(&self, _dur: Option<Duration>, _kind: c_int) -> io::Result<()> { Err(generic_error()) }
    pub fn timeout(&self, _kind: c_int) -> io::Result<Option<Duration>> { Err(generic_error()) }
    pub fn shutdown(&self, _how: Shutdown) -> io::Result<()> { Err(generic_error()) }
    pub fn set_nodelay(&self, _nodelay: bool) -> io::Result<()> { Err(generic_error()) }
    pub fn nodelay(&self) -> io::Result<bool> { Err(generic_error()) }
    pub fn set_nonblocking(&self, _nonblocking: bool) -> io::Result<()> { Err(generic_error()) }
    pub fn take_error(&self) -> io::Result<Option<io::Error>> { Err(generic_error()) }
}


impl AsInner<c_int> for Socket {
    fn as_inner(&self) -> &c_int { self.0.as_inner() }
}

impl FromInner<c_int> for Socket {
    fn from_inner(fd: c_int) -> Socket { Socket(FileDesc::new(fd)) }
}

impl IntoInner<c_int> for Socket {
    fn into_inner(self) -> c_int { self.0.into_raw() }
}
