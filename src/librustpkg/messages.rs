// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use extra::term;
use std::io;

pub fn note(msg: &str) {
    pretty_message(msg, "note: ", term::color::GREEN);
}

pub fn warn(msg: &str) {
    pretty_message(msg, "warning: ", term::color::YELLOW);
}

pub fn error(msg: &str) {
    pretty_message(msg, "error: ", term::color::RED);
}

fn pretty_message<'a>(msg: &'a str,
                      prefix: &'a str,
                      color: term::color::Color) {
    let mut term = term::Terminal::new(io::stdout());
    let mut stdout = io::stdout();
    match term {
        Ok(ref mut t) => {
            t.fg(color);
            t.write(prefix.as_bytes());
            t.reset();
        },
        _ => {
            stdout.write(prefix.as_bytes());
        }
    }
    stdout.write(msg.as_bytes());
    stdout.write(['\n' as u8]);
}
