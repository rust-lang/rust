use std::fmt;
use std::panic::RefUnwindSafe;

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
