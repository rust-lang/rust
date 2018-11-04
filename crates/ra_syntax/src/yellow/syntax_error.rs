use std::fmt;

use crate::TextRange;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SyntaxError {
    pub kind: SyntaxErrorKind,
    pub range: TextRange,
}

impl SyntaxError {
    pub fn new(kind: SyntaxErrorKind, range: TextRange) -> SyntaxError {
        SyntaxError { kind, range }
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
