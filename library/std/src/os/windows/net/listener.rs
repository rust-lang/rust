use crate::os::windows::io::{AsRawSocket, FromRawSocket, IntoRawSocket, RawSocket};
use crate::os::windows::net::{SocketAddr, UnixStream};
use crate::path::Path;
use crate::sys::net::Socket;
use crate::{io, mem};
#[cfg(windows)]
use crate::{
    os::windows::net::socketaddr_un,
    sys::{
        c::{AF_UNIX, SOCK_STREAM, SOCKADDR_UN, bind, getsockname, listen},
        winsock::startup,
    },
};
pub struct UnixListener(pub Socket);
impl UnixListener {
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<UnixListener> {
        let (addr, len) = socketaddr_un(path.as_ref())?;
        startup();
        let inner = Socket::new(AF_UNIX as _, SOCK_STREAM)?;
        unsafe {
            if bind(inner.as_raw(), &raw const addr as *const _, len as _) != 0 {
                return Err(io::Error::last_os_error());
            }
            if listen(inner.as_raw(), 128) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(UnixListener(inner))
        }
    }
    pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        let mut storage = SOCKADDR_UN::default();
        let mut len = size_of_val(&storage) as _;
        let inner = self.0.accept(&raw mut storage as *mut _, &raw mut len)?;
        let addr = SocketAddr::from_parts(storage, len as _)?;
        Ok((UnixStream(inner), addr))
    }
    pub fn bind_addr(socket_addr: &SocketAddr) -> io::Result<UnixListener> {
        startup();
        let inner = Socket::new(AF_UNIX as _, SOCK_STREAM)?;
        unsafe {
            if bind(inner.as_raw(), &raw const socket_addr.addr as *const _, socket_addr.len as _)
                != 0
            {
                return Err(io::Error::last_os_error());
            }
            if listen(inner.as_raw(), 128) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(UnixListener(inner))
        }
    }
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        SocketAddr::new(|addr, len| unsafe {
            getsockname(self.0.as_raw() as _, addr, len as _) as _
        })
    }
    pub fn try_clone(&self) -> io::Result<UnixListener> {
        self.0.duplicate().map(UnixListener)
    }
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.0.set_nonblocking(nonblocking)
    }
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0.take_error()
    }
    pub fn incoming(&self) -> Incoming<'_> {
        Incoming { listener: self }
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
