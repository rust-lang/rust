//! Terminal formatting module.
//!
//! This module provides the `Terminal` trait, which abstracts over an [ANSI
//! Terminal][ansi] to provide color printing, among other things. There are two
//! implementations, the `TerminfoTerminal`, which uses control characters from
//! a [terminfo][ti] database, and `WinConsole`, which uses the [Win32 Console
//! API][win].
//!
//! [ansi]: https://en.wikipedia.org/wiki/ANSI_escape_code
//! [win]: https://docs.microsoft.com/en-us/windows/console/character-mode-applications
//! [ti]: https://en.wikipedia.org/wiki/Terminfo

#![deny(missing_docs)]

use std::io::prelude::*;
use std::io::{self};

pub(crate) use terminfo::TerminfoTerminal;
#[cfg(windows)]
pub(crate) use win::WinConsole;

pub(crate) mod terminfo;

#[cfg(windows)]
mod win;

/// Alias for stdout terminals.
pub(crate) type StdoutTerminal = dyn Terminal + Send;

#[cfg(not(windows))]
/// Returns a Terminal wrapping stdout, or None if a terminal couldn't be
/// opened.
pub(crate) fn stdout() -> Option<Box<StdoutTerminal>> {
    TerminfoTerminal::new(io::stdout()).map(|t| Box::new(t) as Box<StdoutTerminal>)
}

#[cfg(windows)]
/// Returns a Terminal wrapping stdout, or None if a terminal couldn't be
/// opened.
pub(crate) fn stdout() -> Option<Box<StdoutTerminal>> {
    TerminfoTerminal::new(io::stdout())
        .map(|t| Box::new(t) as Box<StdoutTerminal>)
        .or_else(|| Some(Box::new(WinConsole::new(io::stdout())) as Box<StdoutTerminal>))
}

/// Terminal color definitions
#[allow(missing_docs)]
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) mod color {
    /// Number for a terminal color
    pub(crate) type Color = u32;

    pub(crate) const BLACK: Color = 0;
    pub(crate) const RED: Color = 1;
    pub(crate) const GREEN: Color = 2;
    pub(crate) const YELLOW: Color = 3;
    pub(crate) const BLUE: Color = 4;
    pub(crate) const MAGENTA: Color = 5;
    pub(crate) const CYAN: Color = 6;
    pub(crate) const WHITE: Color = 7;
}

/// A terminal with similar capabilities to an ANSI Terminal
/// (foreground/background colors etc).
pub trait Terminal: Write {
    /// Sets the foreground color to the given color.
    ///
    /// If the color is a bright color, but the terminal only supports 8 colors,
    /// the corresponding normal color will be used instead.
    ///
    /// Returns `Ok(true)` if the color was set, `Ok(false)` otherwise, and `Err(e)`
    /// if there was an I/O error.
    fn fg(&mut self, color: color::Color) -> io::Result<bool>;

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
}
