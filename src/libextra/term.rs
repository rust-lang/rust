// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple ANSI color library

#[allow(missing_doc)];

use core::prelude::*;

use core::io;

#[cfg(not(target_os = "win32"))] use core::os;
#[cfg(not(target_os = "win32"))] use terminfo::*;
#[cfg(not(target_os = "win32"))] use terminfo::searcher::open;
#[cfg(not(target_os = "win32"))] use terminfo::parser::compiled::parse;
#[cfg(not(target_os = "win32"))] use terminfo::parm::{expand, Number, Variables};

// FIXME (#2807): Windows support.

pub mod color {
    pub type Color = u16;

    pub static black:   Color = 0u16;
    pub static red:     Color = 1u16;
    pub static green:   Color = 2u16;
    pub static yellow:  Color = 3u16;
    pub static blue:    Color = 4u16;
    pub static magenta: Color = 5u16;
    pub static cyan:    Color = 6u16;
    pub static white:   Color = 7u16;

    pub static bright_black:   Color = 8u16;
    pub static bright_red:     Color = 9u16;
    pub static bright_green:   Color = 10u16;
    pub static bright_yellow:  Color = 11u16;
    pub static bright_blue:    Color = 12u16;
    pub static bright_magenta: Color = 13u16;
    pub static bright_cyan:    Color = 14u16;
    pub static bright_white:   Color = 15u16;
}

#[cfg(not(target_os = "win32"))]
pub struct Terminal {
    num_colors: u16,
    priv out: @io::Writer,
    priv ti: ~TermInfo
}

#[cfg(target_os = "win32")]
pub struct Terminal {
    num_colors: u16,
    priv out: @io::Writer,
}

#[cfg(not(target_os = "win32"))]
impl Terminal {
    pub fn new(out: @io::Writer) -> Result<Terminal, ~str> {
        let term = os::getenv("TERM");
        if term.is_none() {
            return Err(~"TERM environment variable undefined");
        }

        let entry = open(term.unwrap());
        if entry.is_err() {
            return Err(entry.unwrap_err());
        }

        let ti = parse(entry.unwrap(), false);
        if ti.is_err() {
            return Err(ti.unwrap_err());
        }

        let inf = ti.unwrap();
        let nc = if inf.strings.find_equiv(&("setaf")).is_some()
                 && inf.strings.find_equiv(&("setab")).is_some() {
                     inf.numbers.find_equiv(&("colors")).map_consume_default(0, |&n| n)
                 } else { 0 };

        return Ok(Terminal {out: out, ti: inf, num_colors: nc});
    }
    /// Sets the foreground color to the given color.
    ///
    /// If the color is a bright color, but the terminal only supports 8 colors,
    /// the corresponding normal color will be used instead.
    pub fn fg(&self, color: color::Color) {
        let color = self.dim_if_necessary(color);
        if self.num_colors > color {
            let s = expand(*self.ti.strings.find_equiv(&("setaf")).unwrap(),
                           [Number(color as int)], &mut Variables::new());
            if s.is_ok() {
                self.out.write(s.unwrap());
            } else {
                warn!("%s", s.unwrap_err());
            }
        }
    }
    /// Sets the background color to the given color.
    ///
    /// If the color is a bright color, but the terminal only supports 8 colors,
    /// the corresponding normal color will be used instead.
    pub fn bg(&self, color: color::Color) {
        let color = self.dim_if_necessary(color);
        if self.num_colors > color {
            let s = expand(*self.ti.strings.find_equiv(&("setab")).unwrap(),
                           [Number(color as int)], &mut Variables::new());
            if s.is_ok() {
                self.out.write(s.unwrap());
            } else {
                warn!("%s", s.unwrap_err());
            }
        }
    }
    pub fn reset(&self) {
        let mut vars = Variables::new();
        let s = do self.ti.strings.find_equiv(&("op"))
                       .map_consume_default(Err(~"can't find op")) |&op| {
                           expand(op, [], &mut vars)
                       };
        if s.is_ok() {
            self.out.write(s.unwrap());
        } else {
            warn!("%s", s.unwrap_err());
        }
    }

    priv fn dim_if_necessary(&self, color: color::Color) -> color::Color {
        if color >= self.num_colors && color >= 8 && color < 16 {
            color-8
        } else { color }
    }
}

#[cfg(target_os = "win32")]
impl Terminal {
    pub fn new(out: @io::Writer) -> Result<Terminal, ~str> {
        return Ok(Terminal {out: out, num_colors: 0});
    }

    pub fn fg(&self, _color: color::Color) {
    }

    pub fn bg(&self, _color: color::Color) {
    }

    pub fn reset(&self) {
    }
}
