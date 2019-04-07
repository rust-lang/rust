//! Terminfo database interface.

use std::collections::HashMap;
use std::env;
use std::error;
use std::fmt;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::path::Path;

use crate::Attr;
use crate::color;
use crate::Terminal;

use searcher::get_dbpath_for_term;
use parser::compiled::{parse, msys_terminfo};
use parm::{expand, Variables, Param};

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

/// A terminfo creation error.
#[derive(Debug)]
pub enum Error {
    /// TermUnset Indicates that the environment doesn't include enough information to find
    /// the terminfo entry.
    TermUnset,
    /// MalformedTerminfo indicates that parsing the terminfo entry failed.
    MalformedTerminfo(String),
    /// io::Error forwards any io::Errors encountered when finding or reading the terminfo entry.
    IoError(io::Error),
}

impl error::Error for Error {
    fn description(&self) -> &str {
        "failed to create TermInfo"
    }

    fn cause(&self) -> Option<&dyn error::Error> {
        use Error::*;
        match *self {
            IoError(ref e) => Some(e),
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
    pub fn from_env() -> Result<TermInfo, Error> {
        let term = match env::var("TERM") {
            Ok(name) => TermInfo::from_name(&name),
            Err(..) => return Err(Error::TermUnset),
        };

        if term.is_err() && env::var("MSYSCON").ok().map_or(false, |s| "mintty.exe" == s) {
            // msys terminal
            Ok(msys_terminfo())
        } else {
            term
        }
    }

    /// Creates a TermInfo for the named terminal.
    pub fn from_name(name: &str) -> Result<TermInfo, Error> {
        get_dbpath_for_term(name)
            .ok_or_else(|| {
                Error::IoError(io::Error::new(io::ErrorKind::NotFound, "terminfo file not found"))
            })
            .and_then(|p| TermInfo::from_path(&(*p)))
    }

    /// Parse the given TermInfo.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<TermInfo, Error> {
        Self::_from_path(path.as_ref())
    }
    // Keep the metadata small
    fn _from_path(path: &Path) -> Result<TermInfo, Error> {
        let file = File::open(path).map_err(Error::IoError)?;
        let mut reader = BufReader::new(file);
        parse(&mut reader, false).map_err(Error::MalformedTerminfo)
    }
}

pub mod searcher;

/// TermInfo format parsing.
pub mod parser {
    //! ncurses-compatible compiled terminfo format parsing (term(5))
    pub mod compiled;
}
pub mod parm;


fn cap_for_attr(attr: Attr) -> &'static str {
    match attr {
        Attr::Bold => "bold",
        Attr::Dim => "dim",
        Attr::Italic(true) => "sitm",
        Attr::Italic(false) => "ritm",
        Attr::Underline(true) => "smul",
        Attr::Underline(false) => "rmul",
        Attr::Blink => "blink",
        Attr::Standout(true) => "smso",
        Attr::Standout(false) => "rmso",
        Attr::Reverse => "rev",
        Attr::Secure => "invis",
        Attr::ForegroundColor(_) => "setaf",
        Attr::BackgroundColor(_) => "setab",
    }
}

/// A Terminal that knows how many colors it supports, with a reference to its
/// parsed Terminfo database record.
pub struct TerminfoTerminal<T> {
    num_colors: u16,
    out: T,
    ti: TermInfo,
}

impl<T: Write + Send> Terminal for TerminfoTerminal<T> {
    type Output = T;
    fn fg(&mut self, color: color::Color) -> io::Result<bool> {
        let color = self.dim_if_necessary(color);
        if self.num_colors > color {
            return self.apply_cap("setaf", &[Param::Number(color as i32)]);
        }
        Ok(false)
    }

    fn bg(&mut self, color: color::Color) -> io::Result<bool> {
        let color = self.dim_if_necessary(color);
        if self.num_colors > color {
            return self.apply_cap("setab", &[Param::Number(color as i32)]);
        }
        Ok(false)
    }

    fn attr(&mut self, attr: Attr) -> io::Result<bool> {
        match attr {
            Attr::ForegroundColor(c) => self.fg(c),
            Attr::BackgroundColor(c) => self.bg(c),
            _ => self.apply_cap(cap_for_attr(attr), &[]),
        }
    }

    fn supports_attr(&self, attr: Attr) -> bool {
        match attr {
            Attr::ForegroundColor(_) | Attr::BackgroundColor(_) => self.num_colors > 0,
            _ => {
                let cap = cap_for_attr(attr);
                self.ti.strings.get(cap).is_some()
            }
        }
    }

    fn reset(&mut self) -> io::Result<bool> {
        // are there any terminals that have color/attrs and not sgr0?
        // Try falling back to sgr, then op
        let cmd = match ["sgr0", "sgr", "op"]
                            .iter()
                            .filter_map(|cap| self.ti.strings.get(*cap))
                            .next() {
            Some(op) => {
                match expand(&op, &[], &mut Variables::new()) {
                    Ok(cmd) => cmd,
                    Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
                }
            }
            None => return Ok(false),
        };
        self.out.write_all(&cmd).and(Ok(true))
    }

    fn get_ref(&self) -> &T {
        &self.out
    }

    fn get_mut(&mut self) -> &mut T {
        &mut self.out
    }

    fn into_inner(self) -> T
        where Self: Sized
    {
        self.out
    }
}

impl<T: Write + Send> TerminfoTerminal<T> {
    /// Creates a new TerminfoTerminal with the given TermInfo and Write.
    pub fn new_with_terminfo(out: T, terminfo: TermInfo) -> TerminfoTerminal<T> {
        let nc = if terminfo.strings.contains_key("setaf") &&
                    terminfo.strings.contains_key("setab") {
            terminfo.numbers.get("colors").map_or(0, |&n| n)
        } else {
            0
        };

        TerminfoTerminal {
            out,
            ti: terminfo,
            num_colors: nc,
        }
    }

    /// Creates a new TerminfoTerminal for the current environment with the given Write.
    ///
    /// Returns `None` when the terminfo cannot be found or parsed.
    pub fn new(out: T) -> Option<TerminfoTerminal<T>> {
        TermInfo::from_env().map(move |ti| TerminfoTerminal::new_with_terminfo(out, ti)).ok()
    }

    fn dim_if_necessary(&self, color: color::Color) -> color::Color {
        if color >= self.num_colors && color >= 8 && color < 16 {
            color - 8
        } else {
            color
        }
    }

    fn apply_cap(&mut self, cmd: &str, params: &[Param]) -> io::Result<bool> {
        match self.ti.strings.get(cmd) {
            Some(cmd) => {
                match expand(&cmd, params, &mut Variables::new()) {
                    Ok(s) => self.out.write_all(&s).and(Ok(true)),
                    Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e)),
                }
            }
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
