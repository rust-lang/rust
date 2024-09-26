//! Terminfo database interface.

use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::{env, error, fmt, io};

use parm::{Param, Variables, expand};
use parser::compiled::{msys_terminfo, parse};
use searcher::get_dbpath_for_term;

use super::{Terminal, color};

/// A parsed terminfo database entry.
#[allow(unused)]
#[derive(Debug)]
pub(crate) struct TermInfo {
    /// Names for the terminal
    pub(crate) names: Vec<String>,
    /// Map of capability name to boolean value
    pub(crate) bools: HashMap<String, bool>,
    /// Map of capability name to numeric value
    pub(crate) numbers: HashMap<String, u32>,
    /// Map of capability name to raw (unexpanded) string
    pub(crate) strings: HashMap<String, Vec<u8>>,
}

/// A terminfo creation error.
#[derive(Debug)]
pub(crate) enum Error {
    /// TermUnset Indicates that the environment doesn't include enough information to find
    /// the terminfo entry.
    TermUnset,
    /// MalformedTerminfo indicates that parsing the terminfo entry failed.
    MalformedTerminfo(String),
    /// io::Error forwards any io::Errors encountered when finding or reading the terminfo entry.
    IoError(io::Error),
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        use Error::*;
        match self {
            IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Error::*;
        match *self {
            TermUnset => Ok(()),
            MalformedTerminfo(ref e) => e.fmt(f),
            IoError(ref e) => e.fmt(f),
        }
    }
}

impl TermInfo {
    /// Creates a TermInfo based on current environment.
    pub(crate) fn from_env() -> Result<TermInfo, Error> {
        let term = match env::var("TERM") {
            Ok(name) => TermInfo::from_name(&name),
            Err(..) => return Err(Error::TermUnset),
        };

        if term.is_err() && env::var("MSYSCON").map_or(false, |s| "mintty.exe" == s) {
            // msys terminal
            Ok(msys_terminfo())
        } else {
            term
        }
    }

    /// Creates a TermInfo for the named terminal.
    pub(crate) fn from_name(name: &str) -> Result<TermInfo, Error> {
        if cfg!(miri) {
            // Avoid all the work of parsing the terminfo (it's pretty slow under Miri), and just
            // assume that the standard color codes work (like e.g. the 'colored' crate).
            return Ok(TermInfo {
                names: Default::default(),
                bools: Default::default(),
                numbers: Default::default(),
                strings: Default::default(),
            });
        }

        get_dbpath_for_term(name)
            .ok_or_else(|| {
                Error::IoError(io::Error::new(io::ErrorKind::NotFound, "terminfo file not found"))
            })
            .and_then(|p| TermInfo::from_path(&(*p)))
    }

    /// Parse the given TermInfo.
    pub(crate) fn from_path<P: AsRef<Path>>(path: P) -> Result<TermInfo, Error> {
        Self::_from_path(path.as_ref())
    }
    // Keep the metadata small
    fn _from_path(path: &Path) -> Result<TermInfo, Error> {
        let mut reader = File::open_buffered(path).map_err(Error::IoError)?;
        parse(&mut reader, false).map_err(Error::MalformedTerminfo)
    }
}

pub(crate) mod searcher;

/// TermInfo format parsing.
pub(crate) mod parser {
    //! ncurses-compatible compiled terminfo format parsing (term(5))
    pub(crate) mod compiled;
}
pub(crate) mod parm;

/// A Terminal that knows how many colors it supports, with a reference to its
/// parsed Terminfo database record.
pub(crate) struct TerminfoTerminal<T> {
    num_colors: u32,
    out: T,
    ti: TermInfo,
}

impl<T: Write + Send> Terminal for TerminfoTerminal<T> {
    fn fg(&mut self, color: color::Color) -> io::Result<bool> {
        let color = self.dim_if_necessary(color);
        if cfg!(miri) && color < 8 {
            // The Miri logic for this only works for the most basic 8 colors, which we just assume
            // the terminal will support. (`num_colors` is always 0 in Miri, so higher colors will
            // just fail. But libtest doesn't use any higher colors anyway.)
            return write!(self.out, "\x1B[3{color}m").and(Ok(true));
        }
        if self.num_colors > color {
            return self.apply_cap("setaf", &[Param::Number(color as i32)]);
        }
        Ok(false)
    }

    fn reset(&mut self) -> io::Result<bool> {
        if cfg!(miri) {
            return write!(self.out, "\x1B[0m").and(Ok(true));
        }
        // are there any terminals that have color/attrs and not sgr0?
        // Try falling back to sgr, then op
        let cmd = match ["sgr0", "sgr", "op"].iter().find_map(|cap| self.ti.strings.get(*cap)) {
            Some(op) => match expand(op, &[], &mut Variables::new()) {
                Ok(cmd) => cmd,
                Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
            },
            None => return Ok(false),
        };
        self.out.write_all(&cmd).and(Ok(true))
    }
}

impl<T: Write + Send> TerminfoTerminal<T> {
    /// Creates a new TerminfoTerminal with the given TermInfo and Write.
    pub(crate) fn new_with_terminfo(out: T, terminfo: TermInfo) -> TerminfoTerminal<T> {
        let nc = if terminfo.strings.contains_key("setaf") && terminfo.strings.contains_key("setab")
        {
            terminfo.numbers.get("colors").map_or(0, |&n| n)
        } else {
            0
        };

        TerminfoTerminal { out, ti: terminfo, num_colors: nc }
    }

    /// Creates a new TerminfoTerminal for the current environment with the given Write.
    ///
    /// Returns `None` when the terminfo cannot be found or parsed.
    pub(crate) fn new(out: T) -> Option<TerminfoTerminal<T>> {
        TermInfo::from_env().map(move |ti| TerminfoTerminal::new_with_terminfo(out, ti)).ok()
    }

    fn dim_if_necessary(&self, color: color::Color) -> color::Color {
        if color >= self.num_colors && (8..16).contains(&color) { color - 8 } else { color }
    }

    fn apply_cap(&mut self, cmd: &str, params: &[Param]) -> io::Result<bool> {
        match self.ti.strings.get(cmd) {
            Some(cmd) => match expand(cmd, params, &mut Variables::new()) {
                Ok(s) => self.out.write_all(&s).and(Ok(true)),
                Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e)),
            },
            None => Ok(false),
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
