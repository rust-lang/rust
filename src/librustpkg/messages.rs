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
use core::io;
use core::result::*;

pub fn note(msg: &str) {
    pretty_message(msg, "note: ", term::color_green, io::stdout())
}

pub fn warn(msg: &str) {
    pretty_message(msg, "warning: ", term::color_yellow, io::stdout())
}

pub fn error(msg: &str) {
    pretty_message(msg, "error: ", term::color_red, io::stdout())
}

fn pretty_message<'a>(msg: &'a str, prefix: &'a str, color: u8, out: @io::Writer) {
    let term = term::Terminal::new(out);
    match term {
        Ok(ref t) => {
            t.fg(color);
            out.write_str(prefix);
            t.reset();
        },
        _ => {
            out.write_str(prefix);
        }
    }
    out.write_line(msg);
}
