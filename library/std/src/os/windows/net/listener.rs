#![unstable(feature = "windows_unix_domain_sockets", issue = "56533")]

use core::mem;

use super::sockaddr_un;
use crate::io;
use crate::os::raw::c_int;
use crate::os::windows::io::{AsRawSocket, FromRawSocket, IntoRawSocket, RawSocket};
use crate::os::windows::net::{SocketAddr, UnixStream, from_sockaddr_un};
use crate::path::Path;
use crate::sys::c::{self, AF_UNIX, SOCK_STREAM, bind, getsockname, listen};
use crate::sys::net::Socket;
use crate::sys::winsock::startup;
pub struct UnixListener(Socket);

impl UnixListener {
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<UnixListener> {
        unsafe {
            startup();
            let inner = Socket::new(AF_UNIX as i32, SOCK_STREAM)?;
            let (addr, len) = sockaddr_un(path.as_ref())?;
            if bind(inner.as_raw(), &addr as *const _ as *const _, len) != 0 {
                panic!("err: {}", io::Error::last_os_error())
            }
            if listen(inner.as_raw(), 128) != 0 {
                panic!("err: {}", io::Error::last_os_error())
            }
            Ok(UnixListener(inner))
        }
    }
    pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        let mut storage: c::sockaddr_un = unsafe { mem::zeroed() };
        let mut len = mem::size_of_val(&storage) as c_int;
        let sock = self.0.accept(&mut storage as *mut _ as *mut _, &mut len)?;
        let addr = from_sockaddr_un(storage, len)?;
        Ok((UnixStream(sock), addr))
    }
    pub fn incoming(&self) -> Incoming<'_> {
        Incoming { listener: self }
    }
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe { getsockname(self.0.as_raw() as _, addr, len) })
    }
    pub fn try_clone(&self) -> io::Result<UnixListener> {
        self.0.duplicate().map(UnixListener)
    }
}

pub struct Incoming<'a> {
    listener: &'a UnixListener,
}

impl<'a> Iterator for Incoming<'a> {
    type Item = io::Result<UnixStream>;

    fn next(&mut self) -> Option<io::Result<UnixStream>> {
        Some(self.listener.accept().map(|s| s.0))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

impl AsRawSocket for UnixListener {
    fn as_raw_socket(&self) -> RawSocket {
        self.0.as_raw_socket()
    }
}

impl FromRawSocket for UnixListener {
    unsafe fn from_raw_socket(sock: RawSocket) -> Self {
        UnixListener(unsafe { Socket::from_raw_socket(sock) })
    }
}

impl IntoRawSocket for UnixListener {
    fn into_raw_socket(self) -> RawSocket {
        let ret = self.0.as_raw_socket();
        mem::forget(self);
        ret
    }
}

impl<'a> IntoIterator for &'a UnixListener {
    type Item = io::Result<UnixStream>;
    type IntoIter = Incoming<'a>;

    fn into_iter(self) -> Incoming<'a> {
        self.incoming()
    }
}
