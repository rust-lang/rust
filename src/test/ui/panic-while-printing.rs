// run-pass
// ignore-emscripten no subprocess support

#![feature(set_stdio)]

use std::fmt;
use std::fmt::{Display, Formatter};
use std::io::{self, set_panic, LocalOutput, Write};

pub struct A;

impl Display for A {
    fn fmt(&self, _f: &mut Formatter<'_>) -> fmt::Result {
        panic!();
    }
}

struct Sink;
impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl LocalOutput for Sink {
    fn clone_box(&self) -> Box<dyn LocalOutput> {
        Box::new(Sink)
    }
}

fn main() {
    set_panic(Some(Box::new(Sink)));
    assert!(std::panic::catch_unwind(|| {
        eprintln!("{}", A);
    })
    .is_err());
}
