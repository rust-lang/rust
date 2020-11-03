// run-pass
// ignore-emscripten no subprocess support

#![feature(set_stdio)]

use std::fmt;
use std::fmt::{Display, Formatter};
use std::io::{self, set_panic, Write};
use std::sync::{Arc, Mutex};

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

fn main() {
    set_panic(Some(Arc::new(Mutex::new(Sink))));
    assert!(std::panic::catch_unwind(|| {
        eprintln!("{}", A);
    })
    .is_err());
}
