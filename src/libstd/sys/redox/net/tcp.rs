// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::{Error, ErrorKind, Result};
use net::{SocketAddr, Shutdown};
use path::Path;
use sys::fs::{File, OpenOptions};
use sys_common::{AsInner, FromInner, IntoInner};
use time::Duration;
use vec::Vec;

use super::{path_to_peer_addr, path_to_local_addr};

#[derive(Debug)]
pub struct TcpStream(File);

impl TcpStream {
    pub fn connect(addr: &SocketAddr) -> Result<TcpStream> {
        let path = format!("tcp:{}", addr);
        let mut options = OpenOptions::new();
        options.read(true);
        options.write(true);
        Ok(TcpStream(File::open(&Path::new(path.as_str()), &options)?))
    }

    pub fn duplicate(&self) -> Result<TcpStream> {
        Ok(TcpStream(self.0.dup(&[])?))
    }

    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.0.read(buf)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> Result<usize> {
        self.0.read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> Result<usize> {
        self.0.write(buf)
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
        Err(Error::new(ErrorKind::Other, "TcpStream::ttl not implemented"))
    }

    pub fn read_timeout(&self) -> Result<Option<Duration>> {
        Err(Error::new(ErrorKind::Other, "TcpStream::read_timeout not implemented"))
    }

    pub fn write_timeout(&self) -> Result<Option<Duration>> {
        Err(Error::new(ErrorKind::Other, "TcpStream::write_timeout not implemented"))
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

    pub fn set_ttl(&self, _ttl: u32) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpStream::set_ttl not implemented"))
    }

    pub fn set_read_timeout(&self, _dur: Option<Duration>) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpStream::set_read_timeout not implemented"))
    }

    pub fn set_write_timeout(&self, _dur: Option<Duration>) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpStream::set_write_timeout not implemented"))
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
    pub fn bind(addr: &SocketAddr) -> Result<TcpListener> {
        let path = format!("tcp:/{}", addr);
        let mut options = OpenOptions::new();
        options.read(true);
        options.write(true);
        Ok(TcpListener(File::open(&Path::new(path.as_str()), &options)?))
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
        Err(Error::new(ErrorKind::Other, "TcpListener::ttl not implemented"))
    }

    pub fn set_nonblocking(&self, _nonblocking: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpListener::set_nonblocking not implemented"))
    }

    pub fn set_only_v6(&self, _only_v6: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpListener::set_only_v6 not implemented"))
    }

    pub fn set_ttl(&self, _ttl: u32) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "TcpListener::set_ttl not implemented"))
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
