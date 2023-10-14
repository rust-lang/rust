#![unstable(issue = "none", feature = "std_internals")]

use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};

pub type AnonPipe = Box<dyn AnonPipeApi>;

pub trait AnonPipeApi {
    fn read(&self, _buf: &mut [u8]) -> io::Result<usize>;
    fn read_buf(&self, _buf: BorrowedCursor<'_>) -> io::Result<()>;
    fn read_vectored(&self, _bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize>;
    fn is_read_vectored(&self) -> bool;
    fn read_to_end(&self, _buf: &mut Vec<u8>) -> io::Result<usize>;
    fn write(&self, _buf: &[u8]) -> io::Result<usize>;
    fn write_vectored(&self, _bufs: &[IoSlice<'_>]) -> io::Result<usize>;
    fn is_write_vectored(&self) -> bool;
    fn diverge(&self) -> !;
}

pub fn read2(_p1: AnonPipe, _v1: &mut Vec<u8>, _p2: AnonPipe, _v2: &mut Vec<u8>) -> io::Result<()> {
    todo!()
}
