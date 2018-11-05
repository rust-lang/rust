use std::fmt;

use crate::{TextRange, TextUnit};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SyntaxError {
    kind: SyntaxErrorKind,
    location: Location,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Location {
    Offset(TextUnit),
    Range(TextRange),
}

impl Into<Location> for TextUnit {
    fn into(self) -> Location {
        Location::Offset(self)
    }
}

impl Into<Location> for TextRange {
    fn into(self) -> Location {
        Location::Range(self)
    }
}

impl SyntaxError {
    pub fn new<L: Into<Location>>(kind: SyntaxErrorKind, loc: L) -> SyntaxError {
        SyntaxError { kind, location: loc.into() }
    }

    pub fn location(&self) -> Location {
        self.location.clone()
    }

    pub fn offset(&self) -> TextUnit {
        match self.location {
            Location::Offset(offset) => offset,
            Location::Range(range) => range.start(),
        }
    }

    pub fn add_offset(mut self, plus_offset: TextUnit) -> SyntaxError {
        self.location = match self.location {
            Location::Range(range) => Location::Range(range + plus_offset),
            Location::Offset(offset) => Location::Offset(offset + plus_offset)
        };

        self
    }
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.kind.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SyntaxErrorKind {
    ParseError(ParseError),
    EmptyChar,
    UnclosedChar,
    LongChar,
    EmptyAsciiEscape,
    InvalidAsciiEscape,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParseError(pub String);

impl fmt::Display for SyntaxErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::SyntaxErrorKind::*;
        match self {
            EmptyAsciiEscape => write!(f, "Empty escape sequence"),
            InvalidAsciiEscape => write!(f, "Invalid escape sequence"),
            EmptyChar => write!(f, "Empty char literal"),
            UnclosedChar => write!(f, "Unclosed char literal"),
            LongChar => write!(f, "Char literal should be one character long"),
            ParseError(msg) => write!(f, "{}", msg.0),
        }
    }
}
