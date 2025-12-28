//! Protocol framing

use std::io::{self, BufRead, Write};

pub trait Framing {
    type Buf: Default + Send + Sync;

    fn read<'a, R: BufRead + ?Sized>(
        inp: &mut R,
        buf: &'a mut Self::Buf,
    ) -> io::Result<Option<&'a mut Self::Buf>>;

    fn write<W: Write + ?Sized>(out: &mut W, buf: &Self::Buf) -> io::Result<()>;
}
