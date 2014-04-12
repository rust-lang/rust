// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple ANSI color library

#![crate_id = "term#0.11-pre"]
#![comment = "Simple ANSI color library"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]

#![feature(macro_rules)]

#![deny(missing_doc)]

extern crate collections;

use std::io;
use std::os;
use terminfo::TermInfo;
use terminfo::searcher::open;
use terminfo::parser::compiled::{parse, msys_terminfo};
use terminfo::parm::{expand, Number, Variables};

pub mod terminfo;

// FIXME (#2807): Windows support.

/// Terminal color definitions
pub mod color {
    /// Number for a terminal color
    pub type Color = u16;

    pub static BLACK:   Color = 0u16;
    pub static RED:     Color = 1u16;
    pub static GREEN:   Color = 2u16;
    pub static YELLOW:  Color = 3u16;
    pub static BLUE:    Color = 4u16;
    pub static MAGENTA: Color = 5u16;
    pub static CYAN:    Color = 6u16;
    pub static WHITE:   Color = 7u16;

    pub static BRIGHT_BLACK:   Color = 8u16;
    pub static BRIGHT_RED:     Color = 9u16;
    pub static BRIGHT_GREEN:   Color = 10u16;
    pub static BRIGHT_YELLOW:  Color = 11u16;
    pub static BRIGHT_BLUE:    Color = 12u16;
    pub static BRIGHT_MAGENTA: Color = 13u16;
    pub static BRIGHT_CYAN:    Color = 14u16;
    pub static BRIGHT_WHITE:   Color = 15u16;
}

/// Terminal attributes
pub mod attr {
    /// Terminal attributes for use with term.attr().
    ///
    /// Most attributes can only be turned on and must be turned off with term.reset().
    /// The ones that can be turned off explicitly take a boolean value.
    /// Color is also represented as an attribute for convenience.
    pub enum Attr {
        /// Bold (or possibly bright) mode
        Bold,
        /// Dim mode, also called faint or half-bright. Often not supported
        Dim,
        /// Italics mode. Often not supported
        Italic(bool),
        /// Underline mode
        Underline(bool),
        /// Blink mode
        Blink,
        /// Standout mode. Often implemented as Reverse, sometimes coupled with Bold
        Standout(bool),
        /// Reverse mode, inverts the foreground and background colors
        Reverse,
        /// Secure mode, also called invis mode. Hides the printed text
        Secure,
        /// Convenience attribute to set the foreground color
        ForegroundColor(super::color::Color),
        /// Convenience attribute to set the background color
        BackgroundColor(super::color::Color)
    }
}

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
/// parsed TermInfo database record.
pub struct Terminal<T> {
    num_colors: u16,
    out: T,
    ti: ~TermInfo
}

impl<T: Writer> Terminal<T> {
    /// Returns a wrapped output stream (`Terminal<T>`) as a `Result`.
    ///
    /// Returns `Err()` if the TERM environment variable is undefined.
    /// TERM should be set to something like `xterm-color` or `screen-256color`.
    ///
    /// Returns `Err()` on failure to open the terminfo database correctly.
    /// Also, in the event that the individual terminfo database entry can not
    /// be parsed.
    pub fn new(out: T) -> Result<Terminal<T>, ~str> {
        let term = match os::getenv("TERM") {
            Some(t) => t,
            None => return Err(~"TERM environment variable undefined")
        };

        let mut file = match open(term) {
            Ok(file) => file,
            Err(err) => {
                if "cygwin" == term { // msys terminal
                    return Ok(Terminal {
                        out: out,
                        ti: msys_terminfo(),
                        num_colors: 8
                    });
                }
                return Err(err);
            }
        };

        let inf = try!(parse(&mut file, false));

        let nc = if inf.strings.find_equiv(&("setaf")).is_some()
                 && inf.strings.find_equiv(&("setab")).is_some() {
                     inf.numbers.find_equiv(&("colors")).map_or(0, |&n| n)
                 } else { 0 };

        return Ok(Terminal {out: out, ti: inf, num_colors: nc});
    }
    /// Sets the foreground color to the given color.
    ///
    /// If the color is a bright color, but the terminal only supports 8 colors,
    /// the corresponding normal color will be used instead.
    ///
    /// Returns `Ok(true)` if the color was set, `Ok(false)` otherwise, and `Err(e)`
    /// if there was an I/O error.
    pub fn fg(&mut self, color: color::Color) -> io::IoResult<bool> {
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
    /// Sets the background color to the given color.
    ///
    /// If the color is a bright color, but the terminal only supports 8 colors,
    /// the corresponding normal color will be used instead.
    ///
    /// Returns `Ok(true)` if the color was set, `Ok(false)` otherwise, and `Err(e)`
    /// if there was an I/O error.
    pub fn bg(&mut self, color: color::Color) -> io::IoResult<bool> {
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

    /// Sets the given terminal attribute, if supported.
    /// Returns `Ok(true)` if the attribute was supported, `Ok(false)` otherwise,
    /// and `Err(e)` if there was an I/O error.
    pub fn attr(&mut self, attr: attr::Attr) -> io::IoResult<bool> {
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

    /// Returns whether the given terminal attribute is supported.
    pub fn supports_attr(&self, attr: attr::Attr) -> bool {
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

    /// Resets all terminal attributes and color to the default.
    /// Returns `Ok()`.
    pub fn reset(&mut self) -> io::IoResult<()> {
        let mut cap = self.ti.strings.find_equiv(&("sgr0"));
        if cap.is_none() {
            // are there any terminals that have color/attrs and not sgr0?
            // Try falling back to sgr, then op
            cap = self.ti.strings.find_equiv(&("sgr"));
            if cap.is_none() {
                cap = self.ti.strings.find_equiv(&("op"));
            }
        }
        let s = cap.map_or(Err(~"can't find terminfo capability `sgr0`"), |op| {
            expand(op.as_slice(), [], &mut Variables::new())
        });
        if s.is_ok() {
            return self.out.write(s.unwrap().as_slice())
        }
        Ok(())
    }

    fn dim_if_necessary(&self, color: color::Color) -> color::Color {
        if color >= self.num_colors && color >= 8 && color < 16 {
            color-8
        } else { color }
    }

    /// Returns the contained stream
    pub fn unwrap(self) -> T { self.out }

    /// Gets an immutable reference to the stream inside
    pub fn get_ref<'a>(&'a self) -> &'a T { &self.out }

    /// Gets a mutable reference to the stream inside
    pub fn get_mut<'a>(&'a mut self) -> &'a mut T { &mut self.out }
}

impl<T: Writer> Writer for Terminal<T> {
    fn write(&mut self, buf: &[u8]) -> io::IoResult<()> {
        self.out.write(buf)
    }

    fn flush(&mut self) -> io::IoResult<()> {
        self.out.flush()
    }
}
