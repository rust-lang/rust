#![unstable(feature = "windows_unix_domain_sockets", issue = "56533")]

use core::mem;
use core::time::Duration;

use crate::io::{self, IoSlice};
use crate::net::Shutdown;
use crate::os::windows::io::{
    AsRawSocket, AsSocket, BorrowedSocket, FromRawSocket, IntoRawSocket, RawSocket,
};
use crate::os::windows::net::{SocketAddr, sockaddr_un};
use crate::path::Path;
use crate::sys::c::{SO_RCVTIMEO, SO_SNDTIMEO, connect, getpeername, getsockname};
use crate::sys::net::Socket;
use crate::sys::winsock::startup;
pub struct UnixStream(pub Socket);
impl UnixStream {
    pub fn connect<P: AsRef<Path>>(path: P) -> io::Result<UnixStream> {
        unsafe {
            startup();
            let inner = Socket::new_unix()?;
            let (addr, len) = sockaddr_un(path.as_ref())?;
            if connect(inner.as_raw() as _, &addr as *const _ as *const _, len) != 0 {
                panic!("err: {}", io::Error::last_os_error())
            }
            Ok(UnixStream(inner))
        }
    }
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { getsockname(self.0.as_raw() as _, addr, len) })
    }
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { getpeername(self.0.as_raw() as _, addr, len) })
    }
    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(SO_RCVTIMEO)
    }
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }
    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(dur, SO_RCVTIMEO)
    }
    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.0.set_timeout(dur, SO_SNDTIMEO)
    }
    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        self.0.shutdown(how)
    }
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }
    pub fn try_clone(&self) -> io::Result<UnixStream> {
        self.0.duplicate().map(UnixStream)
    }
    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0.timeout(SO_SNDTIMEO)
    }
}

impl io::Read for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        io::Read::read(&mut &*self, buf)
    }
}

impl<'a> io::Read for &'a UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
}

impl io::Write for UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        io::Write::write(&mut &*self, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::Write::flush(&mut &*self)
    }
}
impl<'a> io::Write for &'a UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write_vectored(&[IoSlice::new(buf)])
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl AsSocket for UnixStream {
    fn as_socket(&self) -> BorrowedSocket<'_> {
        self.0.as_socket()
    }
}

impl AsRawSocket for UnixStream {
    fn as_raw_socket(&self) -> RawSocket {
        self.0.as_raw_socket()
    }
}

impl FromRawSocket for UnixStream {
    unsafe fn from_raw_socket(sock: RawSocket) -> Self {
        unsafe { UnixStream(Socket::from_raw_socket(sock)) }
    }
}

impl IntoRawSocket for UnixStream {
    fn into_raw_socket(self) -> RawSocket {
        let ret = self.0.as_raw_socket();
        mem::forget(self);
        ret
    }
}
