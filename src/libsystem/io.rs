use error::prelude::*;

use core::fmt;

pub mod traits {
    pub use super::{Read as sys_Read, Write as sys_Write, Seek as sys_Seek};
}

pub mod prelude {
    pub use super::{FmtWrite, SeekFrom, Seek, Read, Write};
}

pub enum SeekFrom {
    Start(u64),
    End(i64),
    Current(i64),
}

pub trait Read {
    fn read(&self, data: &mut [u8]) -> Result<usize>;
}

pub trait Write {
    fn write(&self, data: &[u8]) -> Result<usize>;

    fn flush(&self) -> Result<()> { Ok(()) }

    fn by_ref(&self) -> &Self where Self: Sized { self }
    fn fmt_writer(self) -> FmtWrite<Self> where Self: Sized { FmtWrite(self) }
}

impl<'a, W: Write> Write for &'a W {
    fn write(&self, data: &[u8]) -> Result<usize> { (*self).write(data) }

    fn flush(&self) -> Result<()> { (*self).flush() }
}

pub trait Seek {
    fn seek(&self, pos: SeekFrom) -> Result<u64>;
}

pub struct FmtWrite<W>(W);

impl<W: Write> fmt::Write for FmtWrite<W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0.write(s.as_bytes()).map_err(|_| fmt::Error).map(drop)
    }
}
