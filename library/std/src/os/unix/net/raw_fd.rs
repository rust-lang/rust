use crate::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
use crate::sys_common::{self, AsInner, FromInner, IntoInner};
use crate::{net, sys};

macro_rules! impl_as_raw_fd {
    ($($t:ident)*) => {$(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl AsRawFd for net::$t {
            fn as_raw_fd(&self) -> RawFd {
                *self.as_inner().socket().as_inner()
            }
        }
    )*};
}
impl_as_raw_fd! { TcpStream TcpListener UdpSocket }

macro_rules! impl_from_raw_fd {
    ($($t:ident)*) => {$(
        #[stable(feature = "from_raw_os", since = "1.1.0")]
        impl FromRawFd for net::$t {
            unsafe fn from_raw_fd(fd: RawFd) -> net::$t {
                let socket = sys::net::Socket::from_inner(fd);
                net::$t::from_inner(sys_common::net::$t::from_inner(socket))
            }
        }
    )*};
}
impl_from_raw_fd! { TcpStream TcpListener UdpSocket }

macro_rules! impl_into_raw_fd {
    ($($t:ident)*) => {$(
        #[stable(feature = "into_raw_os", since = "1.4.0")]
        impl IntoRawFd for net::$t {
            fn into_raw_fd(self) -> RawFd {
                self.into_inner().into_socket().into_inner()
            }
        }
    )*};
}
impl_into_raw_fd! { TcpStream TcpListener UdpSocket }
