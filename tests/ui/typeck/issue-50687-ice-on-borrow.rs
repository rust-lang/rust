// This previously caused an ICE at:
// librustc/traits/structural_impls.rs:180: impossible case reached

#![no_main]

use std::borrow::Borrow;
use std::io;
use std::io::Write;

trait Constraint {}

struct Container<T> {
    t: T,
}

struct Borrowed;
struct Owned;

impl<'a, T> Write for &'a Container<T>
where
    T: Constraint,
    &'a T: Write,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Borrow<Borrowed> for Owned {
    fn borrow(&self) -> &Borrowed {
        &Borrowed
    }
}

fn func(owned: Owned) {
    let _: () = Borrow::borrow(&owned); //~ ERROR mismatched types
}
