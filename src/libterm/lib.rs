// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Terminal text decoration library

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
#[cfg(windows)] extern crate libc;

use std::io;
use std::os;
use terminfo::TermInfo;
use terminfo::searcher::open;
use terminfo::parser::compiled::{parse, msys_terminfo};
use terminfo::parm::{expand, Number, Variables};

pub mod terminfo;

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

struct Terminal<T> {
    num_colors: u16,
    out: T,
    ti: ~TermInfo
}

/// Terminal operations
pub trait TerminalOps<T: Writer> : Writer {
    /// Sets the foreground color to the given color.
    ///
    /// If the color is a bright color, but the terminal only supports 8 colors,
    /// the corresponding normal color will be used instead.
    ///
    /// Returns `Ok(true)` if the color was set, `Ok(false)` otherwise, and `Err(e)`
    /// if there was an I/O error.
    fn fg(&mut self, color: color::Color) -> io::IoResult<bool>;

    /// Sets the background color to the given color.
    ///
    /// If the color is a bright color, but the terminal only supports 8 colors,
    /// the corresponding normal color will be used instead.
    ///
    /// Returns `Ok(true)` if the color was set, `Ok(false)` otherwise, and `Err(e)`
    /// if there was an I/O error.
    fn bg(&mut self, color: color::Color) -> io::IoResult<bool>;

    /// Sets the given terminal attribute, if supported.
    /// Returns `Ok(true)` if the attribute was supported, `Ok(false)` otherwise,
    /// and `Err(e)` if there was an I/O error.
    fn attr(&mut self, attr: attr::Attr) -> io::IoResult<bool>;

    /// Returns whether the given terminal attribute is supported.
    fn supports_attr(&self, attr: attr::Attr) -> bool;

    /// Resets all terminal attributes and color to the default.
    /// Returns `Ok()`.
    fn reset(&mut self) -> io::IoResult<()>;

    /// Gets an immutable reference to the stream inside
    fn get_ref<'a>(&'a self) -> &'a T;

    /// Gets a mutable reference to the stream inside
    fn get_mut<'a>(&'a mut self) -> &'a mut T;
}

/// Returns a wrapped output stream (`~TerminalOps<T>`) as a `Result`.
///
/// Returns `Err()` if the TERM environment variable is undefined.
/// TERM should be set to something like `xterm-color` or `screen-256color`.
///
/// Returns `Err()` on failure to open the terminfo database correctly.
/// Also, in the event that the individual terminfo database entry can not
/// be parsed.
///
/// On Windows, if the TERM environment variable is undefined, or the
/// variable is defined and is set to "cygwin", it assumes the program is
/// running on Windows console and returns platform-specific implementation.
/// Otherwise, it will behave like other platforms.
pub fn new_terminal<T: Writer + Send>(out: T) -> Result<~TerminalOps<T>:Send, ~str> {
    new_terminal_platform(out)
}

#[cfg(not(windows))]
fn new_terminal_platform<T: Writer + Send>(out: T) -> Result<~TerminalOps<T>:Send, ~str> {
    let term = match os::getenv("TERM") {
        Some(t) => t,
        None => return Err("TERM environment variable undefined".to_owned())
    };
    open_terminal(out, term)
}

#[cfg(windows)]
fn new_terminal_platform<T: Writer + Send>(out: T) -> Result<~TerminalOps<T>:Send, ~str> {
    match os::getenv("TERM") {
        Some(term) => {
            if term == "cygwin".to_owned() { // msys terminal
                windows::new_console(out)
            } else {
                open_terminal(out, term)
            }
        },
        None => windows::new_console(out)
    }
}

fn open_terminal<T: Writer + Send>(out: T, term: ~str) -> Result<~TerminalOps<T>:Send, ~str> {
    let mut file = match open(term) {
        Ok(file) => file,
        Err(err) => {
            if "xterm" == term { // MSYS rxvt
                return Ok(box Terminal {
                    out: out,
                    ti: msys_terminfo(),
                    num_colors: 8
                } as ~TerminalOps<T>:Send);
            }
            return Err(err);
        }
    };

    let inf = try!(parse(&mut file, false));

    let nc = if inf.strings.find_equiv(&("setaf")).is_some()
             && inf.strings.find_equiv(&("setab")).is_some() {
                 inf.numbers.find_equiv(&("colors")).map_or(0, |&n| n)
             } else { 0 };

    return Ok(box Terminal {out: out, ti: inf, num_colors: nc} as ~TerminalOps<T>:Send);
}

impl<T: Writer> TerminalOps<T> for Terminal<T> {
    fn fg(&mut self, color: color::Color) -> io::IoResult<bool> {
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

    fn bg(&mut self, color: color::Color) -> io::IoResult<bool> {
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

    fn attr(&mut self, attr: attr::Attr) -> io::IoResult<bool> {
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

    fn reset(&mut self) -> io::IoResult<()> {
        let mut cap = self.ti.strings.find_equiv(&("sgr0"));
        if cap.is_none() {
            // are there any terminals that have color/attrs and not sgr0?
            // Try falling back to sgr, then op
            cap = self.ti.strings.find_equiv(&("sgr"));
            if cap.is_none() {
                cap = self.ti.strings.find_equiv(&("op"));
            }
        }
        let s = cap.map_or(Err("can't find terminfo capability `sgr0`".to_owned()), |op| {
            expand(op.as_slice(), [], &mut Variables::new())
        });
        if s.is_ok() {
            return self.out.write(s.unwrap().as_slice())
        }
        Ok(())
    }

    fn get_ref<'a>(&'a self) -> &'a T { &self.out }

    fn get_mut<'a>(&'a mut self) -> &'a mut T { &mut self.out }
}

impl<T: Writer> Terminal<T> {
    fn dim_if_necessary(&self, color: color::Color) -> color::Color {
        if color >= self.num_colors && color >= 8 && color < 16 {
            color-8
        } else { color }
    }
}

impl<T: Writer> Writer for Terminal<T> {
    fn write(&mut self, buf: &[u8]) -> io::IoResult<()> {
        self.out.write(buf)
    }

    fn flush(&mut self) -> io::IoResult<()> {
        self.out.flush()
    }
}

#[cfg(windows)]
#[allow(non_camel_case_types)]
#[allow(uppercase_variables)]
mod windows {
    use std::io;
    use std::os::win32::as_utf16_p;
    use std::ptr;
    use libc::{BOOL, HANDLE, WORD};
    use libc::{GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING,
               FILE_ATTRIBUTE_NORMAL, INVALID_HANDLE_VALUE};
    use libc::{CreateFileW, CloseHandle};

    use color;
    use attr;

    use TerminalOps;
    #[cfg(test)] use new_terminal;

    pub struct WindowsConsole<T> {
        out: T,
        conout: HANDLE,
        orig_attr: WORD
    }

    // FOREGROUND_* and BACKGROUND_* color attributes have same bit pattern.
    // Constants below can be used in place of FOREGROUND values, and shift the value
    // left by four bits to get BACKGROUNDs.
    static BLUE: WORD = 1;
    static GREEN: WORD = 2;
    static RED: WORD = 4;
    static INTENSITY: WORD = 8;

    static CLEAR_FG_COLOR_BITS_MASK: WORD = 0xfff0;
    static CLEAR_BG_COLOR_BITS_MASK: WORD = 0xff0f;

    type SHORT = i16;
    struct SMALL_RECT {
        Left: SHORT,
        Top: SHORT,
        Right: SHORT,
        Bottom: SHORT
    }
    struct COORD {
        X: SHORT,
        Y: SHORT
    }
    struct CONSOLE_SCREEN_BUFFER_INFO {
        dwSize: COORD,
        dwCursorPosition: COORD,
        wAttributes: WORD,
        srWindow: SMALL_RECT,
        dwMaximumWindowSize: COORD
    }
    type PCONSOLE_SCREEN_BUFFER_INFO = *mut CONSOLE_SCREEN_BUFFER_INFO;

    extern "system" {
        fn GetConsoleScreenBufferInfo(
                hConsoleOutput: HANDLE,
                lpConsoleScreenBufferInfo: PCONSOLE_SCREEN_BUFFER_INFO
                ) -> BOOL;
        fn SetConsoleTextAttribute(hConsoleOutput: HANDLE,
                                   wAttributes: WORD) -> BOOL;
    }

    fn get_console_attr(console: HANDLE) -> Option<WORD> {
        let mut info = CONSOLE_SCREEN_BUFFER_INFO {
            dwSize: COORD { X: 0, Y: 0 },
            dwCursorPosition: COORD { X: 0, Y: 0 },
            wAttributes: 0,
            srWindow: SMALL_RECT { Left: 0, Top: 0, Right: 0, Bottom: 0 },
            dwMaximumWindowSize: COORD { X: 0, Y: 0 }
        };
        if unsafe { GetConsoleScreenBufferInfo(console, &mut info) } != 0 {
            Some(info.wAttributes)
        } else {
            None
        }
    }

    fn set_console_attr(console: HANDLE, value: WORD) -> bool {
        unsafe { SetConsoleTextAttribute(console, value) != 0 }
    }

    fn color_to_console_attr(color: color::Color) -> WORD {
        match color {
            color::BLACK => 0,
            color::RED => RED,
            color::GREEN => GREEN,
            color::YELLOW => RED | GREEN,
            color::BLUE => BLUE,
            color::MAGENTA => RED | BLUE,
            color::CYAN => GREEN | BLUE,
            color::WHITE => RED | GREEN | BLUE,

            color::BRIGHT_BLACK => INTENSITY,
            color::BRIGHT_RED => RED | INTENSITY,
            color::BRIGHT_GREEN => GREEN | INTENSITY,
            color::BRIGHT_YELLOW => RED | GREEN | INTENSITY,
            color::BRIGHT_BLUE => BLUE | INTENSITY,
            color::BRIGHT_MAGENTA => RED | BLUE | INTENSITY,
            color::BRIGHT_CYAN => GREEN | BLUE | INTENSITY,
            color::BRIGHT_WHITE => RED | GREEN | BLUE | INTENSITY,

            _ => unreachable!()
        }
    }

    fn open_console() -> Option<HANDLE> {
        let conout = as_utf16_p("CONOUT$", |p| unsafe {
            CreateFileW(p, GENERIC_READ | GENERIC_WRITE, 0, ptr::mut_null(),
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, ptr::mut_null())
        });
        if conout as uint == INVALID_HANDLE_VALUE as uint {
            return None
        }
        Some(conout)
    }

    pub fn new_console<T: Writer + Send>(out: T) -> Result<~TerminalOps<T>:Send, ~str> {
        let conout = match open_console() {
            Some(c) => c,
            None => return Err("Cannot open Windows console".to_owned())
        };
        let orig_attr = match get_console_attr(conout) {
            Some(value) => value,
            None => return Err("Couldn't get the current screen buffer attribute.".to_owned())
        };
        let console = WindowsConsole {conout: conout, out: out, orig_attr: orig_attr};
        Ok(box console as ~TerminalOps<T>:Send)
    }

    impl<T: Writer> TerminalOps<T> for WindowsConsole<T> {
        fn fg(&mut self, color: color::Color) -> io::IoResult<bool> {
            let color = color_to_console_attr(color);
            let console_attr = match get_console_attr(self.conout) {
                Some(attr) => attr,
                None => return Ok(false)
            };
            Ok(set_console_attr(self.conout, console_attr & CLEAR_FG_COLOR_BITS_MASK | color))
        }

        fn bg(&mut self, color: color::Color) -> io::IoResult<bool> {
            // shift left to get background color
            let color = color_to_console_attr(color) << 4;
            let console_attr = match get_console_attr(self.conout) {
                Some(attr) => attr,
                None => return Ok(false)
            };
            Ok(set_console_attr(self.conout, console_attr & CLEAR_BG_COLOR_BITS_MASK | color))
        }

        fn attr(&mut self, attr: attr::Attr) -> io::IoResult<bool> {
            match attr {
                attr::ForegroundColor(c) => self.fg(c),
                attr::BackgroundColor(c) => self.bg(c),
                _ => Ok(false)
            }
        }

        fn supports_attr(&self, attr: attr::Attr) -> bool {
            match attr {
                attr::ForegroundColor(_) | attr::BackgroundColor(_) => true,
                _ => false
            }
        }

        fn reset(&mut self) -> io::IoResult<()> {
            set_console_attr(self.conout, self.orig_attr);
            Ok(())
        }

        fn get_ref<'a>(&'a self) -> &'a T { &self.out }

        fn get_mut<'a>(&'a mut self) -> &'a mut T { &mut self.out }
    }

    impl<T: Writer> Writer for WindowsConsole<T> {
        fn write(&mut self, buf: &[u8]) -> io::IoResult<()> {
            // We need to flush immediately after write call because Windows
            // console maintains colors and attributes as a state on API,
            // unlike TTYs, which encode these in characters.
            // We can avoid this by manually keep track of states and writes,
            // although it doesn't seem to worth complexity.
            self.out.write(buf).and_then(|_| self.out.flush())
        }

        fn flush(&mut self) -> io::IoResult<()> {
            self.out.flush()
        }
    }

    #[unsafe_destructor]
    impl<T: Writer> Drop for WindowsConsole<T> {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.conout);
            }
        }
    }

    #[cfg(test)]
    #[test]
    fn test_windows_console() {
        let conout = match open_console() {
            Some(c) => c,
            None => return ()
        };
        let orig_attr = get_console_attr(conout).unwrap();
        let mut t = new_terminal(io::stdout()).unwrap();

        t.fg(color::MAGENTA).unwrap();
        assert_eq!(get_console_attr(conout).unwrap() & 0xf, RED | BLUE);

        t.bg(color::CYAN).unwrap();
        assert_eq!(get_console_attr(conout).unwrap() & 0xff,
                   RED | BLUE | (GREEN | BLUE) << 4);

        t.fg(color::YELLOW).unwrap();
        assert_eq!(get_console_attr(conout).unwrap() & 0xff,
                   RED | GREEN | (GREEN | BLUE) << 4);

        t.reset().unwrap();
        assert_eq!(get_console_attr(conout).unwrap(), orig_attr);

        assert!(t.supports_attr(attr::ForegroundColor(color::RED)));
        assert!(t.supports_attr(attr::BackgroundColor(color::BLUE)));
        assert!(!t.supports_attr(attr::Bold));

        t.attr(attr::ForegroundColor(color::RED)).unwrap();
        assert_eq!(get_console_attr(conout).unwrap() & 0xf, RED);
        t.attr(attr::BackgroundColor(color::BLUE)).unwrap();
        assert_eq!(get_console_attr(conout).unwrap() & 0xff, RED | BLUE << 4);

        t.reset().unwrap();
    }
}
