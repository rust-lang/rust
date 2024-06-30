use crate::{
    fmt,
    io::{self, BorrowedCursor, IoSlice, IoSliceMut},
};

pub struct AnonPipe(!);

impl fmt::Debug for AnonPipe {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

impl AnonPipe {
    pub fn try_clone(&self) -> io::Result<Self> {
        self.0
    }

    pub fn read(&self, _buf: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn read_buf(&self, _buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.0
    }

    pub fn read_vectored(&self, _bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0
    }

    pub fn is_read_vectored(&self) -> bool {
        self.0
    }

    pub fn read_to_end(&self, _buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0
    }

    pub fn write(&self, _buf: &[u8]) -> io::Result<usize> {
        self.0
    }

    pub fn write_vectored(&self, _bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0
    }

    pub fn is_write_vectored(&self) -> bool {
        self.0
    }

    pub fn diverge(&self) -> ! {
        self.0
    }
}

pub fn read2(p1: AnonPipe, _v1: &mut Vec<u8>, _p2: AnonPipe, _v2: &mut Vec<u8>) -> io::Result<()> {
    match p1.0 {}
}
