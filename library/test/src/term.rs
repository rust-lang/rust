//! Terminal formatting module.
//!
//! This module provides the `Terminal` trait, which abstracts over an [ANSI
//! Terminal][ansi] to provide color printing, among other things. There are two
//! implementations, the `TerminfoTerminal`, which uses control characters from
//! a [terminfo][ti] database, and `WinConsole`, which uses the [Win32 Console
//! API][win].
//!
//! ```
//!
//! [ansi]: https://en.wikipedia.org/wiki/ANSI_escape_code
//! [win]: https://docs.microsoft.com/en-us/windows/console/character-mode-applications
//! [ti]: https://en.wikipedia.org/wiki/Terminfo

#![deny(missing_docs)]

use std::io::prelude::*;
use std::io::{self, Stderr, Stdout};

pub use terminfo::TerminfoTerminal;
#[cfg(windows)]
pub use win::WinConsole;

pub mod terminfo;

#[cfg(windows)]
mod win;

/// Alias for stdout terminals.
pub type StdoutTerminal = dyn Terminal<Output = Stdout> + Send;
/// Alias for stderr terminals.
pub type StderrTerminal = dyn Terminal<Output = Stderr> + Send;

#[cfg(not(windows))]
/// Returns a Terminal wrapping stdout, or None if a terminal couldn't be
/// opened.
pub fn stdout() -> Option<Box<StdoutTerminal>> {
    TerminfoTerminal::new(io::stdout()).map(|t| Box::new(t) as Box<StdoutTerminal>)
}

#[cfg(windows)]
/// Returns a Terminal wrapping stdout, or None if a terminal couldn't be
/// opened.
pub fn stdout() -> Option<Box<StdoutTerminal>> {
    TerminfoTerminal::new(io::stdout())
        .map(|t| Box::new(t) as Box<StdoutTerminal>)
        .or_else(|| WinConsole::new(io::stdout()).ok().map(|t| Box::new(t) as Box<StdoutTerminal>))
}

#[cfg(not(windows))]
/// Returns a Terminal wrapping stderr, or None if a terminal couldn't be
/// opened.
pub fn stderr() -> Option<Box<StderrTerminal>> {
    TerminfoTerminal::new(io::stderr()).map(|t| Box::new(t) as Box<StderrTerminal>)
}

#[cfg(windows)]
/// Returns a Terminal wrapping stderr, or None if a terminal couldn't be
/// opened.
pub fn stderr() -> Option<Box<StderrTerminal>> {
    TerminfoTerminal::new(io::stderr())
        .map(|t| Box::new(t) as Box<StderrTerminal>)
        .or_else(|| WinConsole::new(io::stderr()).ok().map(|t| Box::new(t) as Box<StderrTerminal>))
}

/// Terminal color definitions
#[allow(missing_docs)]
pub mod color {
    /// Number for a terminal color
    pub type Color = u32;

    pub const BLACK: Color = 0;
    pub const RED: Color = 1;
    pub const GREEN: Color = 2;
    pub const YELLOW: Color = 3;
    pub const BLUE: Color = 4;
    pub const MAGENTA: Color = 5;
    pub const CYAN: Color = 6;
    pub const WHITE: Color = 7;

    pub const BRIGHT_BLACK: Color = 8;
    pub const BRIGHT_RED: Color = 9;
    pub const BRIGHT_GREEN: Color = 10;
    pub const BRIGHT_YELLOW: Color = 11;
    pub const BRIGHT_BLUE: Color = 12;
    pub const BRIGHT_MAGENTA: Color = 13;
    pub const BRIGHT_CYAN: Color = 14;
    pub const BRIGHT_WHITE: Color = 15;
}

/// Terminal attributes for use with term.attr().
///
/// Most attributes can only be turned on and must be turned off with term.reset().
/// The ones that can be turned off explicitly take a boolean value.
/// Color is also represented as an attribute for convenience.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
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
    ForegroundColor(color::Color),
    /// Convenience attribute to set the background color
    BackgroundColor(color::Color),
}

/// A terminal with similar capabilities to an ANSI Terminal
/// (foreground/background colors etc).
pub trait Terminal: Write {
    /// The terminal's output writer type.
    type Output: Write;

    /// Sets the foreground color to the given color.
    ///
    /// If the color is a bright color, but the terminal only supports 8 colors,
    /// the corresponding normal color will be used instead.
    ///
    /// Returns `Ok(true)` if the color was set, `Ok(false)` otherwise, and `Err(e)`
    /// if there was an I/O error.
    fn fg(&mut self, color: color::Color) -> io::Result<bool>;

    /// Sets the background color to the given color.
    ///
    /// If the color is a bright color, but the terminal only supports 8 colors,
    /// the corresponding normal color will be used instead.
    ///
    /// Returns `Ok(true)` if the color was set, `Ok(false)` otherwise, and `Err(e)`
    /// if there was an I/O error.
    fn bg(&mut self, color: color::Color) -> io::Result<bool>;

    /// Sets the given terminal attribute, if supported. Returns `Ok(true)`
    /// if the attribute was supported, `Ok(false)` otherwise, and `Err(e)` if
    /// there was an I/O error.
    fn attr(&mut self, attr: Attr) -> io::Result<bool>;

    /// Returns `true` if the given terminal attribute is supported.
    fn supports_attr(&self, attr: Attr) -> bool;

    /// Resets all terminal attributes and colors to their defaults.
    ///
    /// Returns `Ok(true)` if the terminal was reset, `Ok(false)` otherwise, and `Err(e)` if there
    /// was an I/O error.
    ///
    /// *Note: This does not flush.*
    ///
    /// That means the reset command may get buffered so, if you aren't planning on doing anything
    /// else that might flush stdout's buffer (e.g., writing a line of text), you should flush after
    /// calling reset.
    fn reset(&mut self) -> io::Result<bool>;

    /// Gets an immutable reference to the stream inside
    fn get_ref(&self) -> &Self::Output;

    /// Gets a mutable reference to the stream inside
    fn get_mut(&mut self) -> &mut Self::Output;

    /// Returns the contained stream, destroying the `Terminal`
    fn into_inner(self) -> Self::Output
    where
        Self: Sized;
}
