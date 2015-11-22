// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Terminfo database interface.

use std::collections::HashMap;
use std::env;
use std::io::prelude::*;
use std::io;

use attr;
use color;
use Terminal;
use UnwrappableTerminal;
use self::searcher::open;
use self::parser::compiled::{parse, msys_terminfo};
use self::parm::{expand, Number, Variables};


/// A parsed terminfo database entry.
#[derive(Debug)]
pub struct TermInfo {
    /// Names for the terminal
    pub names: Vec<String>,
    /// Map of capability name to boolean value
    pub bools: HashMap<String, bool>,
    /// Map of capability name to numeric value
    pub numbers: HashMap<String, u16>,
    /// Map of capability name to raw (unexpanded) string
    pub strings: HashMap<String, Vec<u8>>,
}

pub mod searcher;

/// TermInfo format parsing.
pub mod parser {
    //! ncurses-compatible compiled terminfo format parsing (term(5))
    pub mod compiled;
}
pub mod parm;


fn cap_for_attr(attr: attr::Attr) -> &'static str {
    match attr {
        attr::Bold => "bold",
        attr::Dim => "dim",
        attr::Italic(true) => "sitm",
        attr::Italic(false) => "ritm",
        attr::Underline(true) => "smul",
        attr::Underline(false) => "rmul",
        attr::Blink => "blink",
        attr::Standout(true) => "smso",
        attr::Standout(false) => "rmso",
        attr::Reverse => "rev",
        attr::Secure => "invis",
        attr::ForegroundColor(_) => "setaf",
        attr::BackgroundColor(_) => "setab",
    }
}

/// A Terminal that knows how many colors it supports, with a reference to its
/// parsed Terminfo database record.
pub struct TerminfoTerminal<T> {
    num_colors: u16,
    out: T,
    ti: Box<TermInfo>,
}

impl<T: Write+Send+'static> Terminal<T> for TerminfoTerminal<T> {
    fn fg(&mut self, color: color::Color) -> io::Result<bool> {
        let color = self.dim_if_necessary(color);
        if self.num_colors > color {
            let s = expand(self.ti
                               .strings
                               .get("setaf")
                               .unwrap(),
                           &[Number(color as isize)],
                           &mut Variables::new());
            if s.is_ok() {
                try!(self.out.write_all(&s.unwrap()));
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn bg(&mut self, color: color::Color) -> io::Result<bool> {
        let color = self.dim_if_necessary(color);
        if self.num_colors > color {
            let s = expand(self.ti
                               .strings
                               .get("setab")
                               .unwrap(),
                           &[Number(color as isize)],
                           &mut Variables::new());
            if s.is_ok() {
                try!(self.out.write_all(&s.unwrap()));
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn attr(&mut self, attr: attr::Attr) -> io::Result<bool> {
        match attr {
            attr::ForegroundColor(c) => self.fg(c),
            attr::BackgroundColor(c) => self.bg(c),
            _ => {
                let cap = cap_for_attr(attr);
                let parm = self.ti.strings.get(cap);
                if parm.is_some() {
                    let s = expand(parm.unwrap(), &[], &mut Variables::new());
                    if s.is_ok() {
                        try!(self.out.write_all(&s.unwrap()));
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    fn supports_attr(&self, attr: attr::Attr) -> bool {
        match attr {
            attr::ForegroundColor(_) | attr::BackgroundColor(_) => self.num_colors > 0,
            _ => {
                let cap = cap_for_attr(attr);
                self.ti.strings.get(cap).is_some()
            }
        }
    }

    fn reset(&mut self) -> io::Result<()> {
        let mut cap = self.ti.strings.get("sgr0");
        if cap.is_none() {
            // are there any terminals that have color/attrs and not sgr0?
            // Try falling back to sgr, then op
            cap = self.ti.strings.get("sgr");
            if cap.is_none() {
                cap = self.ti.strings.get("op");
            }
        }
        let s = cap.map_or(Err("can't find terminfo capability `sgr0`".to_owned()),
                           |op| expand(op, &[], &mut Variables::new()));
        if s.is_ok() {
            return self.out.write_all(&s.unwrap());
        }
        Ok(())
    }

    fn get_ref<'a>(&'a self) -> &'a T {
        &self.out
    }

    fn get_mut<'a>(&'a mut self) -> &'a mut T {
        &mut self.out
    }
}

impl<T: Write+Send+'static> UnwrappableTerminal<T> for TerminfoTerminal<T> {
    fn unwrap(self) -> T {
        self.out
    }
}

impl<T: Write+Send+'static> TerminfoTerminal<T> {
    /// Returns `None` whenever the terminal cannot be created for some
    /// reason.
    pub fn new(out: T) -> Option<Box<Terminal<T> + Send + 'static>> {
        let term = match env::var("TERM") {
            Ok(t) => t,
            Err(..) => {
                debug!("TERM environment variable not defined");
                return None;
            }
        };

        let mut file = match open(&term[..]) {
            Ok(f) => f,
            Err(err) => {
                return match env::var("MSYSCON") {
                    Ok(ref val) if &val[..] == "mintty.exe" => {
                        // msys terminal
                        Some(box TerminfoTerminal {
                            out: out,
                            ti: msys_terminfo(),
                            num_colors: 8,
                        })
                    }
                    _ => {
                        debug!("error finding terminfo entry: {:?}", err);
                        None
                    }
                };
            }
        };

        let ti = parse(&mut file, false);
        if ti.is_err() {
            debug!("error parsing terminfo entry: {:?}", ti.err().unwrap());
            return None;
        }

        let inf = ti.unwrap();
        let nc = if inf.strings.get("setaf").is_some() && inf.strings.get("setab").is_some() {
            inf.numbers.get("colors").map_or(0, |&n| n)
        } else {
            0
        };

        Some(box TerminfoTerminal {
            out: out,
            ti: inf,
            num_colors: nc,
        })
    }

    fn dim_if_necessary(&self, color: color::Color) -> color::Color {
        if color >= self.num_colors && color >= 8 && color < 16 {
            color - 8
        } else {
            color
        }
    }
}


impl<T: Write> Write for TerminfoTerminal<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.out.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}
