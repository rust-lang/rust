#![unstable(reason = "not public", issue = "0", feature = "fd")]

use io::{self, Read};
use libc::c_int;
use sys::fs::generic_error;
use sys_common::AsInner;

#[derive(Debug)]
pub struct FileDesc {
    fd: c_int,
}

impl FileDesc {
    pub fn new(fd: c_int) -> FileDesc {
        FileDesc { fd: fd }
    }

    pub fn raw(&self) -> c_int { self.fd }
    pub fn into_raw(self) -> c_int { self.fd}

    pub fn read(&self, _buf: &mut [u8]) -> io::Result<usize> { Err(generic_error()) }
    pub fn read_to_end(&self, _buf: &mut Vec<u8>) -> io::Result<usize> { Err(generic_error()) }
    pub fn write(&self, _buf: &[u8]) -> io::Result<usize> { Err(generic_error()) }
    pub fn set_cloexec(&self) -> io::Result<()> { Err(generic_error()) }
    pub fn set_nonblocking(&self, _nonblocking: bool) -> io::Result<()> { Err(generic_error()) }
    pub fn duplicate(&self) -> io::Result<FileDesc> { Err(generic_error()) }
}

impl<'a> Read for &'a FileDesc {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> { Err(generic_error()) }
    fn read_to_end(&mut self, _buf: &mut Vec<u8>) -> io::Result<usize> { Err(generic_error()) }
}

impl AsInner<c_int> for FileDesc {
    fn as_inner(&self) -> &c_int { &self.fd }
}

impl Drop for FileDesc {
    fn drop(&mut self) {}
}
