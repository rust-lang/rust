// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple ANSI color library

use core::io;
use core::option;
use core::os;

// FIXME (#2807): Windows support.

pub static color_black: u8 = 0u8;
pub static color_red: u8 = 1u8;
pub static color_green: u8 = 2u8;
pub static color_yellow: u8 = 3u8;
pub static color_blue: u8 = 4u8;
pub static color_magenta: u8 = 5u8;
pub static color_cyan: u8 = 6u8;
pub static color_light_gray: u8 = 7u8;
pub static color_light_grey: u8 = 7u8;
pub static color_dark_gray: u8 = 8u8;
pub static color_dark_grey: u8 = 8u8;
pub static color_bright_red: u8 = 9u8;
pub static color_bright_green: u8 = 10u8;
pub static color_bright_yellow: u8 = 11u8;
pub static color_bright_blue: u8 = 12u8;
pub static color_bright_magenta: u8 = 13u8;
pub static color_bright_cyan: u8 = 14u8;
pub static color_bright_white: u8 = 15u8;

pub fn esc(writer: @io::Writer) { writer.write([0x1bu8, '[' as u8]); }

/// Reset the foreground and background colors to default
pub fn reset(writer: @io::Writer) {
    esc(writer);
    writer.write(['0' as u8, 'm' as u8]);
}

/// Returns true if the terminal supports color
pub fn color_supported() -> bool {
    let supported_terms = ~[~"xterm-color", ~"xterm",
                           ~"screen-bce", ~"xterm-256color"];
    return match os::getenv("TERM") {
          option::Some(ref env) => {
            for supported_terms.each |term| {
                if *term == *env { return true; }
            }
            false
          }
          option::None => false
        };
}

pub fn set_color(writer: @io::Writer, first_char: u8, color: u8) {
    assert!((color < 16u8));
    esc(writer);
    let mut color = color;
    if color >= 8u8 { writer.write(['1' as u8, ';' as u8]); color -= 8u8; }
    writer.write([first_char, ('0' as u8) + color, 'm' as u8]);
}

/// Set the foreground color
pub fn fg(writer: @io::Writer, color: u8) {
    return set_color(writer, '3' as u8, color);
}

/// Set the background color
pub fn bg(writer: @io::Writer, color: u8) {
    return set_color(writer, '4' as u8, color);
}
