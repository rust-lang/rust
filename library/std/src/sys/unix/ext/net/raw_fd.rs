use crate::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
use crate::sys_common::{self, AsInner, FromInner, IntoInner};
use crate::{net, sys};

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for net::TcpStream {
    fn as_raw_fd(&self) -> RawFd {
        *self.as_inner().socket().as_inner()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for net::TcpListener {
    fn as_raw_fd(&self) -> RawFd {
        *self.as_inner().socket().as_inner()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for net::UdpSocket {
    fn as_raw_fd(&self) -> RawFd {
        *self.as_inner().socket().as_inner()
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for net::TcpStream {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpStream {
        let socket = sys::net::Socket::from_inner(fd);
        net::TcpStream::from_inner(sys_common::net::TcpStream::from_inner(socket))
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for net::TcpListener {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpListener {
        let socket = sys::net::Socket::from_inner(fd);
        net::TcpListener::from_inner(sys_common::net::TcpListener::from_inner(socket))
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for net::UdpSocket {
    unsafe fn from_raw_fd(fd: RawFd) -> net::UdpSocket {
        let socket = sys::net::Socket::from_inner(fd);
        net::UdpSocket::from_inner(sys_common::net::UdpSocket::from_inner(socket))
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for net::TcpStream {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_socket().into_inner()
    }
}
#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for net::TcpListener {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_socket().into_inner()
    }
}
#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for net::UdpSocket {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_socket().into_inner()
    }
}
