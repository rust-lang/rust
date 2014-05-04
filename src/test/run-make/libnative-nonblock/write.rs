// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;
use std::io::IoResult;
use std::os;

fn run() -> IoResult<()> {
    let mut out = io::stdio::stdout_raw();
    for _ in range(0u, 1024) {
        let mut buf = ['x' as u8, ..1024];
        buf[1023] = '\n' as u8;
        try!(out.write(buf));
    }
    Ok(())
}

fn main() {
    match run() {
        Err(e) => {
            (writeln!(&mut io::stderr(), "Error: {}", e)).unwrap();
            os::set_exit_status(1);
        }
        Ok(()) => ()
    }
}
