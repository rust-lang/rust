// run-fail
// check-run-results

#![allow(unused_imports)]
#![feature(core_panic)]
#![feature(error_in_core)]
#![feature(error_in_panic)]

extern crate core;

use core::error;
use core::panicking::panic_source;

use std::error::Error;

#[derive(Debug)]
struct MyErr {
    super_source: SourceError,
}

#[derive(Debug)]
struct SourceError {}

use std::fmt;
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "my source error message")
    }
}

impl error::Error for MyErr {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.super_source)
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "my source's source error message")
    }
}

impl error::Error for SourceError {}

fn main() {
    let source_error = SourceError {};
    let source = MyErr { super_source: source_error };
    panic_source(format_args!("here's my panic error message"), &source);
}
