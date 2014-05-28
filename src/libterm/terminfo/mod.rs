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

use collections::HashMap;
use std::io::IoResult;
use std::os;

use attr;
use color;
use Terminal;
use self::searcher::open;
use self::parser::compiled::{parse, msys_terminfo};
use self::parm::{expand, Number, Variables};


/// A parsed terminfo database entry.
#[deriving(Show)]
pub struct TermInfo {
    /// Names for the terminal
    pub names: Vec<String> ,
    /// Map of capability name to boolean value
    pub bools: HashMap<String, bool>,
    /// Map of capability name to numeric value
    pub numbers: HashMap<String, u16>,
    /// Map of capability name to raw (unexpanded) string
    pub strings: HashMap<String, Vec<u8> >
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
        attr::Bold               => "bold",
        attr::Dim                => "dim",
        attr::Italic(true)       => "sitm",
        attr::Italic(false)      => "ritm",
        attr::Underline(true)    => "smul",
        attr::Underline(false)   => "rmul",
        attr::Blink              => "blink",
        attr::Standout(true)     => "smso",
        attr::Standout(false)    => "rmso",
        attr::Reverse            => "rev",
        attr::Secure             => "invis",
        attr::ForegroundColor(_) => "setaf",
        attr::BackgroundColor(_) => "setab"
    }
}

/// A Terminal that knows how many colors it supports, with a reference to its
/// parsed Terminfo database record.
pub struct TerminfoTerminal<T> {
    num_colors: u16,
    out: T,
    ti: Box<TermInfo>
}

impl<T: Writer> Terminal<T> for TerminfoTerminal<T> {
    fn new(out: T) -> Option<TerminfoTerminal<T>> {
        let term = match os::getenv("TERM") {
            Some(t) => t,
            None => {
                debug!("TERM environment variable not defined");
                return None;
            }
        };

        let entry = open(term.as_slice());
        if entry.is_err() {
            if os::getenv("MSYSCON").map_or(false, |s| {
                    "mintty.exe" == s.as_slice()
                }) {
                // msys terminal
                return Some(TerminfoTerminal {out: out, ti: msys_terminfo(), num_colors: 8});
            }
            debug!("error finding terminfo entry: {}", entry.err().unwrap());
            return None;
        }

        let mut file = entry.unwrap();
        let ti = parse(&mut file, false);
        if ti.is_err() {
            debug!("error parsing terminfo entry: {}", ti.unwrap_err());
            return None;
        }

        let inf = ti.unwrap();
        let nc = if inf.strings.find_equiv(&("setaf")).is_some()
                 && inf.strings.find_equiv(&("setab")).is_some() {
                     inf.numbers.find_equiv(&("colors")).map_or(0, |&n| n)
                 } else { 0 };

        return Some(TerminfoTerminal {out: out, ti: inf, num_colors: nc});
    }

    fn fg(&mut self, color: color::Color) -> IoResult<bool> {
        let color = self.dim_if_necessary(color);
        if self.num_colors > color {
            let s = expand(self.ti
                               .strings
                               .find_equiv(&("setaf"))
                               .unwrap()
                               .as_slice(),
                           [Number(color as int)], &mut Variables::new());
            if s.is_ok() {
                try!(self.out.write(s.unwrap().as_slice()));
                return Ok(true)
            }
        }
        Ok(false)
    }

    fn bg(&mut self, color: color::Color) -> IoResult<bool> {
        let color = self.dim_if_necessary(color);
        if self.num_colors > color {
            let s = expand(self.ti
                               .strings
                               .find_equiv(&("setab"))
                               .unwrap()
                               .as_slice(),
                           [Number(color as int)], &mut Variables::new());
            if s.is_ok() {
                try!(self.out.write(s.unwrap().as_slice()));
                return Ok(true)
            }
        }
        Ok(false)
    }

    fn attr(&mut self, attr: attr::Attr) -> IoResult<bool> {
        match attr {
            attr::ForegroundColor(c) => self.fg(c),
            attr::BackgroundColor(c) => self.bg(c),
            _ => {
                let cap = cap_for_attr(attr);
                let parm = self.ti.strings.find_equiv(&cap);
                if parm.is_some() {
                    let s = expand(parm.unwrap().as_slice(),
                                   [],
                                   &mut Variables::new());
                    if s.is_ok() {
                        try!(self.out.write(s.unwrap().as_slice()));
                        return Ok(true)
                    }
                }
                Ok(false)
            }
        }
    }

    fn supports_attr(&self, attr: attr::Attr) -> bool {
        match attr {
            attr::ForegroundColor(_) | attr::BackgroundColor(_) => {
                self.num_colors > 0
            }
            _ => {
                let cap = cap_for_attr(attr);
                self.ti.strings.find_equiv(&cap).is_some()
            }
        }
    }

    fn reset(&mut self) -> IoResult<()> {
        let mut cap = self.ti.strings.find_equiv(&("sgr0"));
        if cap.is_none() {
            // are there any terminals that have color/attrs and not sgr0?
            // Try falling back to sgr, then op
            cap = self.ti.strings.find_equiv(&("sgr"));
            if cap.is_none() {
                cap = self.ti.strings.find_equiv(&("op"));
            }
        }
        let s = cap.map_or(Err("can't find terminfo capability `sgr0`".to_string()), |op| {
            expand(op.as_slice(), [], &mut Variables::new())
        });
        if s.is_ok() {
            return self.out.write(s.unwrap().as_slice())
        }
        Ok(())
    }

    fn unwrap(self) -> T { self.out }

    fn get_ref<'a>(&'a self) -> &'a T { &self.out }

    fn get_mut<'a>(&'a mut self) -> &'a mut T { &mut self.out }
}

impl<T: Writer> TerminfoTerminal<T> {
    fn dim_if_necessary(&self, color: color::Color) -> color::Color {
        if color >= self.num_colors && color >= 8 && color < 16 {
            color-8
        } else { color }
    }
}


impl<T: Writer> Writer for TerminfoTerminal<T> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.out.write(buf)
    }

    fn flush(&mut self) -> IoResult<()> {
        self.out.flush()
    }
}

