use crate::cmp;
use crate::io::{self, Error, ErrorKind, Result, IoSlice, IoSliceMut};
use crate::mem;
use crate::net::{SocketAddr, Shutdown};
use crate::path::Path;
use crate::sys::fs::{File, OpenOptions};
use crate::sys::syscall::TimeSpec;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::time::Duration;

use super::{path_to_peer_addr, path_to_local_addr};

#[derive(Debug)]
pub struct TcpStream(File);

impl TcpStream {
    pub fn connect(addr: Result<&SocketAddr>) -> Result<TcpStream> {
        let path = format!("tcp:{}", addr?);
        let mut options = OpenOptions::new();
        options.read(true);
        options.write(true);
        Ok(TcpStream(File::open(Path::new(path.as_str()), &options)?))
    }

    pub fn connect_timeout(_addr: &SocketAddr, _timeout: Duration) -> Result<TcpStream> {
        Err(Error::new(ErrorKind::Other, "TcpStream::connect_timeout not implemented"))
    }

    pub fn duplicate(&self) -> Result<TcpStream> {
        Ok(TcpStream(self.0.dup(&[])?))
    }

    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.0.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        io::default_read_vectored(|b| self.read(b), bufs)
    }

    pub fn write(&self, buf: &[u8]) -> Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        io::default_write_vectored(|b| self.write(b), bufs)
    }

    pub fn take_error(&self) -> Result<Option<Error>> {
        Ok(None)
    }

    pub fn peer_addr(&self) -> Result<SocketAddr> {
        let path = self.0.path()?;
        Ok(path_to_peer_addr(path.to_str().unwrap_or("")))
    }

    pub fn socket_addr(&self) -> Result<SocketAddr> {
        let path = self.0.path()?;
        Ok(path_to_local_addr(path.to_str().unwrap_or("")))
    }

    pub fn peek(&self, _buf: &mut [u8]) -> Result<usize> {
        Err(Error::new(ErrorKind::Other, "TcpStream::peek not implemented"))
    }

    pub fn shutdown(&self, _how: Shutdown) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpStream::shutdown not implemented"))
    }

    pub fn nodelay(&self) -> Result<bool> {
        Err(Error::new(ErrorKind::Other, "TcpStream::nodelay not implemented"))
    }

    pub fn nonblocking(&self) -> Result<bool> {
        self.0.fd().nonblocking()
    }

    pub fn only_v6(&self) -> Result<bool> {
        Err(Error::new(ErrorKind::Other, "TcpStream::only_v6 not implemented"))
    }

    pub fn ttl(&self) -> Result<u32> {
        let mut ttl = [0];
        let file = self.0.dup(b"ttl")?;
        file.read(&mut ttl)?;
        Ok(ttl[0] as u32)
    }

    pub fn read_timeout(&self) -> Result<Option<Duration>> {
        let mut time = TimeSpec::default();
        let file = self.0.dup(b"read_timeout")?;
        if file.read(&mut time)? >= mem::size_of::<TimeSpec>() {
            Ok(Some(Duration::new(time.tv_sec as u64, time.tv_nsec as u32)))
        } else {
            Ok(None)
        }
    }

    pub fn write_timeout(&self) -> Result<Option<Duration>> {
        let mut time = TimeSpec::default();
        let file = self.0.dup(b"write_timeout")?;
        if file.read(&mut time)? >= mem::size_of::<TimeSpec>() {
            Ok(Some(Duration::new(time.tv_sec as u64, time.tv_nsec as u32)))
        } else {
            Ok(None)
        }
    }

    pub fn set_nodelay(&self, _nodelay: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpStream::set_nodelay not implemented"))
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> Result<()> {
        self.0.fd().set_nonblocking(nonblocking)
    }

    pub fn set_only_v6(&self, _only_v6: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpStream::set_only_v6 not implemented"))
    }

    pub fn set_ttl(&self, ttl: u32) -> Result<()> {
        let file = self.0.dup(b"ttl")?;
        file.write(&[cmp::min(ttl, 255) as u8])?;
        Ok(())
    }

    pub fn set_read_timeout(&self, duration_option: Option<Duration>) -> Result<()> {
        let file = self.0.dup(b"read_timeout")?;
        if let Some(duration) = duration_option {
            if duration.as_secs() == 0 && duration.subsec_nanos() == 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                          "cannot set a 0 duration timeout"));
            }
            file.write(&TimeSpec {
                tv_sec: duration.as_secs() as i64,
                tv_nsec: duration.subsec_nanos() as i32
            })?;
        } else {
            file.write(&[])?;
        }
        Ok(())
    }

    pub fn set_write_timeout(&self, duration_option: Option<Duration>) -> Result<()> {
        let file = self.0.dup(b"write_timeout")?;
        if let Some(duration) = duration_option {
            if duration.as_secs() == 0 && duration.subsec_nanos() == 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                          "cannot set a 0 duration timeout"));
            }
            file.write(&TimeSpec {
                tv_sec: duration.as_secs() as i64,
                tv_nsec: duration.subsec_nanos() as i32
            })?;
        } else {
            file.write(&[])?;
        }
        Ok(())
    }
}

impl AsInner<File> for TcpStream {
    fn as_inner(&self) -> &File { &self.0 }
}

impl FromInner<File> for TcpStream {
    fn from_inner(file: File) -> TcpStream {
        TcpStream(file)
    }
}

impl IntoInner<File> for TcpStream {
    fn into_inner(self) -> File { self.0 }
}

#[derive(Debug)]
pub struct TcpListener(File);

impl TcpListener {
    pub fn bind(addr: Result<&SocketAddr>) -> Result<TcpListener> {
        let path = format!("tcp:/{}", addr?);
        let mut options = OpenOptions::new();
        options.read(true);
        options.write(true);
        Ok(TcpListener(File::open(Path::new(path.as_str()), &options)?))
    }

    pub fn accept(&self) -> Result<(TcpStream, SocketAddr)> {
        let file = self.0.dup(b"listen")?;
        let path = file.path()?;
        let peer_addr = path_to_peer_addr(path.to_str().unwrap_or(""));
        Ok((TcpStream(file), peer_addr))
    }

    pub fn duplicate(&self) -> Result<TcpListener> {
        Ok(TcpListener(self.0.dup(&[])?))
    }

    pub fn take_error(&self) -> Result<Option<Error>> {
        Ok(None)
    }

    pub fn socket_addr(&self) -> Result<SocketAddr> {
        let path = self.0.path()?;
        Ok(path_to_local_addr(path.to_str().unwrap_or("")))
    }

    pub fn nonblocking(&self) -> Result<bool> {
        Err(Error::new(ErrorKind::Other, "TcpListener::nonblocking not implemented"))
    }

    pub fn only_v6(&self) -> Result<bool> {
        Err(Error::new(ErrorKind::Other, "TcpListener::only_v6 not implemented"))
    }

    pub fn ttl(&self) -> Result<u32> {
        let mut ttl = [0];
        let file = self.0.dup(b"ttl")?;
        file.read(&mut ttl)?;
        Ok(ttl[0] as u32)
    }

    pub fn set_nonblocking(&self, _nonblocking: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpListener::set_nonblocking not implemented"))
    }

    pub fn set_only_v6(&self, _only_v6: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpListener::set_only_v6 not implemented"))
    }

    pub fn set_ttl(&self, ttl: u32) -> Result<()> {
        let file = self.0.dup(b"ttl")?;
        file.write(&[cmp::min(ttl, 255) as u8])?;
        Ok(())
    }
}

impl AsInner<File> for TcpListener {
    fn as_inner(&self) -> &File { &self.0 }
}

impl FromInner<File> for TcpListener {
    fn from_inner(file: File) -> TcpListener {
        TcpListener(file)
    }
}

impl IntoInner<File> for TcpListener {
    fn into_inner(self) -> File { self.0 }
}
