//! auxiliary definitions for suggest-borrow-for-generic-arg.rs, to ensure the suggestion works on
//! functions defined in other crates.

use std::io::{self, Read, Write};
use std::iter::Sum;

pub fn write_stuff<W: Write>(mut writer: W) -> io::Result<()> {
    writeln!(writer, "stuff")
}

pub fn read_and_discard<R: Read>(mut reader: R) -> io::Result<()> {
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).map(|_| ())
}

pub fn sum_three<I: IntoIterator>(iter: I) -> <I as IntoIterator>::Item
    where <I as IntoIterator>::Item: Sum
{
    iter.into_iter().take(3).sum()
}
