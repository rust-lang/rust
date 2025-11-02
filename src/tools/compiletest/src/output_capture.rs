use std::fmt;
use std::panic::RefUnwindSafe;
use std::sync::Mutex;

pub trait ConsoleOut: fmt::Debug + RefUnwindSafe {
    fn write_fmt(&self, args: fmt::Arguments<'_>);
}

#[derive(Debug)]
pub(crate) struct Stdout;

impl ConsoleOut for Stdout {
    fn write_fmt(&self, args: fmt::Arguments<'_>) {
        print!("{args}");
    }
}

#[derive(Debug)]
pub(crate) struct Stderr;

impl ConsoleOut for Stderr {
    fn write_fmt(&self, args: fmt::Arguments<'_>) {
        eprint!("{args}");
    }
}

pub(crate) struct CaptureBuf {
    inner: Mutex<String>,
}

impl CaptureBuf {
    pub(crate) fn new() -> Self {
        Self { inner: Mutex::new(String::new()) }
    }

    pub(crate) fn into_inner(self) -> String {
        self.inner.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

impl fmt::Debug for CaptureBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaptureBuf").finish_non_exhaustive()
    }
}

impl ConsoleOut for CaptureBuf {
    fn write_fmt(&self, args: fmt::Arguments<'_>) {
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        <String as fmt::Write>::write_fmt(&mut s, args).unwrap();
    }
}
