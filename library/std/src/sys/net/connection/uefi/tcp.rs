use super::tcp4;
use crate::io;
use crate::net::SocketAddr;
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
}
