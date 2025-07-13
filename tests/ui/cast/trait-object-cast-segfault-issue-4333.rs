//@ run-pass
#![allow(unused_must_use)]

use std::io;

pub fn main() {
    let stdout = &mut io::stdout() as &mut dyn io::Write;
    stdout.write(b"Hello!");
}

// https://github.com/rust-lang/rust/issues/4333
