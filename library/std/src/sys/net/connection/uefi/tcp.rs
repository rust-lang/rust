use super::tcp4;
use crate::io;
use crate::net::SocketAddr;
use crate::ptr::NonNull;
use crate::sys::{helpers, unsupported};
use crate::time::Duration;

pub(crate) enum Tcp {
    V4(tcp4::Tcp4),
}

impl Tcp {
    pub(crate) fn connect(addr: &SocketAddr, timeout: Option<Duration>) -> io::Result<Self> {
        match addr {
            SocketAddr::V4(x) => {
                let temp = tcp4::Tcp4::new()?;
                temp.configure(true, Some(x), None)?;
                temp.connect(timeout)?;
                Ok(Tcp::V4(temp))
            }
            SocketAddr::V6(_) => todo!(),
        }
    }

    pub(crate) fn write(&self, buf: &[u8], timeout: Option<Duration>) -> io::Result<usize> {
        match self {
            Self::V4(client) => client.write(buf, timeout),
        }
    }

    pub(crate) fn read(&self, buf: &mut [u8], timeout: Option<Duration>) -> io::Result<usize> {
        match self {
            Self::V4(client) => client.read(buf, timeout),
        }
    }

    pub(crate) fn ttl(&self) -> io::Result<u32> {
        match self {
            Self::V4(client) => client.get_mode_data().map(|x| x.time_to_live.into()),
        }
    }

    pub(crate) fn nodelay(&self) -> io::Result<bool> {
        match self {
            Self::V4(client) => {
                let temp = client.get_mode_data()?;
                match NonNull::new(temp.control_option) {
                    Some(x) => unsafe { Ok(x.as_ref().enable_nagle.into()) },
                    None => unsupported(),
                }
            }
        }
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        match self {
            Self::V4(client) => client.get_mode_data().map(|x| {
                SocketAddr::new(
                    helpers::ipv4_from_r_efi(x.access_point.remote_address).into(),
                    x.access_point.remote_port,
                )
            }),
        }
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        match self {
            Self::V4(client) => client.get_mode_data().map(|x| {
                SocketAddr::new(
                    helpers::ipv4_from_r_efi(x.access_point.station_address).into(),
                    x.access_point.station_port,
                )
            }),
        }
    }
}
