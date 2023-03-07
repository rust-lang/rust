// run-pass
#![allow(unused_must_use)]
// pretty-expanded FIXME #23616

use std::io;

pub fn main() {
    let stdout = &mut io::stdout() as &mut dyn io::Write;
    stdout.write(b"Hello!");
}
