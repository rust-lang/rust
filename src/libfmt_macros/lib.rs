//! Macro support for format strings
//!
//! These structures are used when parsing format strings for the compiler.
//! Parsing does not happen at runtime: structures of `std::fmt::rt` are
//! generated instead.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       test(attr(deny(warnings))))]

#![feature(nll)]
#![feature(rustc_private)]
#![feature(unicode_internals)]
#![feature(bool_to_option)]

pub use Piece::*;
pub use Position::*;
pub use Alignment::*;
pub use Flag::*;
pub use Count::*;

use std::str;
use std::string;
use std::iter;

use syntax_pos::{InnerSpan, Symbol};

#[derive(Copy, Clone)]
struct InnerOffset(usize);

impl InnerOffset {
    fn to(self, end: InnerOffset) -> InnerSpan {
        InnerSpan::new(self.0, end.0)
    }
}

/// A piece is a portion of the format string which represents the next part
/// to emit. These are emitted as a stream by the `Parser` class.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Piece<'a> {
    /// A literal string which should directly be emitted
    String(&'a str),
    /// This describes that formatting should process the next argument (as
    /// specified inside) for emission.
    NextArgument(Argument<'a>),
}

/// Representation of an argument specification.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Argument<'a> {
    /// Where to find this argument
    pub position: Position,
    /// How to format the argument
    pub format: FormatSpec<'a>,
}

/// Specification for the formatting of an argument in the format string.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FormatSpec<'a> {
    /// Optionally specified character to fill alignment with.
    pub fill: Option<char>,
    /// Optionally specified alignment.
    pub align: Alignment,
    /// Packed version of various flags provided.
    pub flags: u32,
    /// The integer precision to use.
    pub precision: Count,
    /// The span of the precision formatting flag (for diagnostics).
    pub precision_span: Option<InnerSpan>,
    /// The string width requested for the resulting format.
    pub width: Count,
    /// The span of the width formatting flag (for diagnostics).
    pub width_span: Option<InnerSpan>,
    /// The descriptor string representing the name of the format desired for
    /// this argument, this can be empty or any number of characters, although
    /// it is required to be one word.
    pub ty: &'a str,
    /// The span of the descriptor string (for diagnostics).
    pub ty_span: Option<InnerSpan>,
}

/// Enum describing where an argument for a format can be located.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Position {
    /// The argument is implied to be located at an index
    ArgumentImplicitlyIs(usize),
    /// The argument is located at a specific index given in the format
    ArgumentIs(usize),
    /// The argument has a name.
    ArgumentNamed(Symbol),
}

impl Position {
    pub fn index(&self) -> Option<usize> {
        match self {
            ArgumentIs(i) | ArgumentImplicitlyIs(i) => Some(*i),
            _ => None,
        }
    }
}

/// Enum of alignments which are supported.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Alignment {
    /// The value will be aligned to the left.
    AlignLeft,
    /// The value will be aligned to the right.
    AlignRight,
    /// The value will be aligned in the center.
    AlignCenter,
    /// The value will take on a default alignment.
    AlignUnknown,
}

/// Various flags which can be applied to format strings. The meaning of these
/// flags is defined by the formatters themselves.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Flag {
    /// A `+` will be used to denote positive numbers.
    FlagSignPlus,
    /// A `-` will be used to denote negative numbers. This is the default.
    FlagSignMinus,
    /// An alternate form will be used for the value. In the case of numbers,
    /// this means that the number will be prefixed with the supplied string.
    FlagAlternate,
    /// For numbers, this means that the number will be padded with zeroes,
    /// and the sign (`+` or `-`) will precede them.
    FlagSignAwareZeroPad,
    /// For Debug / `?`, format integers in lower-case hexadecimal.
    FlagDebugLowerHex,
    /// For Debug / `?`, format integers in upper-case hexadecimal.
    FlagDebugUpperHex,
}

/// A count is used for the precision and width parameters of an integer, and
/// can reference either an argument or a literal integer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Count {
    /// The count is specified explicitly.
    CountIs(usize),
    /// The count is specified by the argument with the given name.
    CountIsName(Symbol),
    /// The count is specified by the argument at the given index.
    CountIsParam(usize),
    /// The count is implied and cannot be explicitly specified.
    CountImplied,
}

pub struct ParseError {
    pub description: string::String,
    pub note: Option<string::String>,
    pub label: string::String,
    pub span: InnerSpan,
    pub secondary_label: Option<(string::String, InnerSpan)>,
}

/// The parser structure for interpreting the input format string. This is
/// modeled as an iterator over `Piece` structures to form a stream of tokens
/// being output.
///
/// This is a recursive-descent parser for the sake of simplicity, and if
/// necessary there's probably lots of room for improvement performance-wise.
pub struct Parser<'a> {
    input: &'a str,
    cur: iter::Peekable<str::CharIndices<'a>>,
    /// Error messages accumulated during parsing
    pub errors: Vec<ParseError>,
    /// Current position of implicit positional argument pointer
    curarg: usize,
    /// `Some(raw count)` when the string is "raw", used to position spans correctly
    style: Option<usize>,
    /// Start and end byte offset of every successfully parsed argument
    pub arg_places: Vec<InnerSpan>,
    /// Characters that need to be shifted
    skips: Vec<usize>,
    /// Span of the last opening brace seen, used for error reporting
    last_opening_brace: Option<InnerSpan>,
    /// Wether the source string is comes from `println!` as opposed to `format!` or `print!`
    append_newline: bool,
}

impl<'a> Iterator for Parser<'a> {
    type Item = Piece<'a>;

    fn next(&mut self) -> Option<Piece<'a>> {
        if let Some(&(pos, c)) = self.cur.peek() {
            match c {
                '{' => {
                    let curr_last_brace = self.last_opening_brace;
                    let byte_pos = self.to_span_index(pos);
                    self.last_opening_brace = Some(byte_pos.to(InnerOffset(byte_pos.0 + 1)));
                    self.cur.next();
                    if self.consume('{') {
                        self.last_opening_brace = curr_last_brace;

                        Some(String(self.string(pos + 1)))
                    } else {
                        let arg = self.argument();
                        if let Some(end) = self.must_consume('}') {
                            let start = self.to_span_index(pos);
                            let end = self.to_span_index(end + 1);
                            self.arg_places.push(start.to(end));
                        }
                        Some(NextArgument(arg))
                    }
                }
                '}' => {
                    self.cur.next();
                    if self.consume('}') {
                        Some(String(self.string(pos + 1)))
                    } else {
                        let err_pos = self.to_span_index(pos);
                        self.err_with_note(
                            "unmatched `}` found",
                            "unmatched `}`",
                            "if you intended to print `}`, you can escape it using `}}`",
                            err_pos.to(err_pos),
                        );
                        None
                    }
                }
                '\n' => {
                    Some(String(self.string(pos)))
                }
                _ => Some(String(self.string(pos))),
            }
        } else {
            None
        }
    }
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given format string
    pub fn new(
        s: &'a str,
        style: Option<usize>,
        skips: Vec<usize>,
        append_newline: bool,
    ) -> Parser<'a> {
        Parser {
            input: s,
            cur: s.char_indices().peekable(),
            errors: vec![],
            curarg: 0,
            style,
            arg_places: vec![],
            skips,
            last_opening_brace: None,
            append_newline,
        }
    }

    /// Notifies of an error. The message doesn't actually need to be of type
    /// String, but I think it does when this eventually uses conditions so it
    /// might as well start using it now.
    fn err<S1: Into<string::String>, S2: Into<string::String>>(
        &mut self,
        description: S1,
        label: S2,
        span: InnerSpan,
    ) {
        self.errors.push(ParseError {
            description: description.into(),
            note: None,
            label: label.into(),
            span,
            secondary_label: None,
        });
    }

    /// Notifies of an error. The message doesn't actually need to be of type
    /// String, but I think it does when this eventually uses conditions so it
    /// might as well start using it now.
    fn err_with_note<S1: Into<string::String>, S2: Into<string::String>, S3: Into<string::String>>(
        &mut self,
        description: S1,
        label: S2,
        note: S3,
        span: InnerSpan,
    ) {
        self.errors.push(ParseError {
            description: description.into(),
            note: Some(note.into()),
            label: label.into(),
            span,
            secondary_label: None,
        });
    }

    /// Optionally consumes the specified character. If the character is not at
    /// the current position, then the current iterator isn't moved and `false` is
    /// returned, otherwise the character is consumed and `true` is returned.
    fn consume(&mut self, c: char) -> bool {
        self.consume_pos(c).is_some()
    }

    /// Optionally consumes the specified character. If the character is not at
    /// the current position, then the current iterator isn't moved and `None` is
    /// returned, otherwise the character is consumed and the current position is
    /// returned.
    fn consume_pos(&mut self, c: char) -> Option<usize> {
        if let Some(&(pos, maybe)) = self.cur.peek() {
            if c == maybe {
                self.cur.next();
                return Some(pos);
            }
        }
        None
    }

    fn to_span_index(&self, pos: usize) -> InnerOffset {
        let mut pos = pos;
        // This handles the raw string case, the raw argument is the number of #
        // in r###"..."### (we need to add one because of the `r`).
        let raw = self.style.map(|raw| raw + 1).unwrap_or(0);
        for skip in &self.skips {
            if pos > *skip {
                pos += 1;
            } else if pos == *skip && raw == 0 {
                pos += 1;
            } else {
                break;
            }
        }
        InnerOffset(raw + pos + 1)
    }

    /// Forces consumption of the specified character. If the character is not
    /// found, an error is emitted.
    fn must_consume(&mut self, c: char) -> Option<usize> {
        self.ws();

        if let Some(&(pos, maybe)) = self.cur.peek() {
            if c == maybe {
                self.cur.next();
                Some(pos)
            } else {
                let pos = self.to_span_index(pos);
                let description = format!("expected `'}}'`, found `{:?}`", maybe);
                let label = "expected `}`".to_owned();
                let (note, secondary_label) = if c == '}' {
                    (Some("if you intended to print `{`, you can escape it using `{{`".to_owned()),
                     self.last_opening_brace.map(|sp| {
                        ("because of this opening brace".to_owned(), sp)
                     }))
                } else {
                    (None, None)
                };
                self.errors.push(ParseError {
                    description,
                    note,
                    label,
                    span: pos.to(pos),
                    secondary_label,
                });
                None
            }
        } else {
            let description = format!("expected `{:?}` but string was terminated", c);
            // point at closing `"`
            let pos = self.input.len() - if self.append_newline { 1 } else { 0 };
            let pos = self.to_span_index(pos);
            if c == '}' {
                let label = format!("expected `{:?}`", c);
                let (note, secondary_label) = if c == '}' {
                    (Some("if you intended to print `{`, you can escape it using `{{`".to_owned()),
                     self.last_opening_brace.map(|sp| {
                        ("because of this opening brace".to_owned(), sp)
                     }))
                } else {
                    (None, None)
                };
                self.errors.push(ParseError {
                    description,
                    note,
                    label,
                    span: pos.to(pos),
                    secondary_label,
                });
            } else {
                self.err(description, format!("expected `{:?}`", c), pos.to(pos));
            }
            None
        }
    }

    /// Consumes all whitespace characters until the first non-whitespace character
    fn ws(&mut self) {
        while let Some(&(_, c)) = self.cur.peek() {
            if c.is_whitespace() {
                self.cur.next();
            } else {
                break;
            }
        }
    }

    /// Parses all of a string which is to be considered a "raw literal" in a
    /// format string. This is everything outside of the braces.
    fn string(&mut self, start: usize) -> &'a str {
        // we may not consume the character, peek the iterator
        while let Some(&(pos, c)) = self.cur.peek() {
            match c {
                '{' | '}' => {
                    return &self.input[start..pos];
                }
                _ => {
                    self.cur.next();
                }
            }
        }
        &self.input[start..self.input.len()]
    }

    /// Parses an `Argument` structure, or what's contained within braces inside the format string.
    fn argument(&mut self) -> Argument<'a> {
        let pos = self.position();
        let format = self.format();

        // Resolve position after parsing format spec.
        let pos = match pos {
            Some(position) => position,
            None => {
                let i = self.curarg;
                self.curarg += 1;
                ArgumentImplicitlyIs(i)
            }
        };

        Argument {
            position: pos,
            format,
        }
    }

    /// Parses a positional argument for a format. This could either be an
    /// integer index of an argument, a named argument, or a blank string.
    /// Returns `Some(parsed_position)` if the position is not implicitly
    /// consuming a macro argument, `None` if it's the case.
    fn position(&mut self) -> Option<Position> {
        if let Some(i) = self.integer() {
            Some(ArgumentIs(i))
        } else {
            match self.cur.peek() {
                Some(&(_, c)) if rustc_lexer::is_id_start(c) => {
                    Some(ArgumentNamed(Symbol::intern(self.word())))
                }

                // This is an `ArgumentNext`.
                // Record the fact and do the resolution after parsing the
                // format spec, to make things like `{:.*}` work.
                _ => None,
            }
        }
    }

    /// Parses a format specifier at the current position, returning all of the
    /// relevant information in the `FormatSpec` struct.
    fn format(&mut self) -> FormatSpec<'a> {
        let mut spec = FormatSpec {
            fill: None,
            align: AlignUnknown,
            flags: 0,
            precision: CountImplied,
            precision_span: None,
            width: CountImplied,
            width_span: None,
            ty: &self.input[..0],
            ty_span: None,
        };
        if !self.consume(':') {
            return spec;
        }

        // fill character
        if let Some(&(_, c)) = self.cur.peek() {
            match self.cur.clone().nth(1) {
                Some((_, '>')) | Some((_, '<')) | Some((_, '^')) => {
                    spec.fill = Some(c);
                    self.cur.next();
                }
                _ => {}
            }
        }
        // Alignment
        if self.consume('<') {
            spec.align = AlignLeft;
        } else if self.consume('>') {
            spec.align = AlignRight;
        } else if self.consume('^') {
            spec.align = AlignCenter;
        }
        // Sign flags
        if self.consume('+') {
            spec.flags |= 1 << (FlagSignPlus as u32);
        } else if self.consume('-') {
            spec.flags |= 1 << (FlagSignMinus as u32);
        }
        // Alternate marker
        if self.consume('#') {
            spec.flags |= 1 << (FlagAlternate as u32);
        }
        // Width and precision
        let mut havewidth = false;

        if self.consume('0') {
            // small ambiguity with '0$' as a format string. In theory this is a
            // '0' flag and then an ill-formatted format string with just a '$'
            // and no count, but this is better if we instead interpret this as
            // no '0' flag and '0$' as the width instead.
            if self.consume('$') {
                spec.width = CountIsParam(0);
                havewidth = true;
            } else {
                spec.flags |= 1 << (FlagSignAwareZeroPad as u32);
            }
        }
        if !havewidth {
            let width_span_start = if let Some((pos, _)) = self.cur.peek() {
                *pos
            } else {
                0
            };
            let (w, sp) = self.count(width_span_start);
            spec.width = w;
            spec.width_span = sp;
        }
        if let Some(start) = self.consume_pos('.') {
            if let Some(end) = self.consume_pos('*') {
                // Resolve `CountIsNextParam`.
                // We can do this immediately as `position` is resolved later.
                let i = self.curarg;
                self.curarg += 1;
                spec.precision = CountIsParam(i);
                spec.precision_span =
                    Some(self.to_span_index(start).to(self.to_span_index(end + 1)));
            } else {
                let (p, sp) = self.count(start);
                spec.precision = p;
                spec.precision_span = sp;
            }
        }
        let ty_span_start = self.cur.peek().map(|(pos, _)| *pos);
        // Optional radix followed by the actual format specifier
        if self.consume('x') {
            if self.consume('?') {
                spec.flags |= 1 << (FlagDebugLowerHex as u32);
                spec.ty = "?";
            } else {
                spec.ty = "x";
            }
        } else if self.consume('X') {
            if self.consume('?') {
                spec.flags |= 1 << (FlagDebugUpperHex as u32);
                spec.ty = "?";
            } else {
                spec.ty = "X";
            }
        } else if self.consume('?') {
            spec.ty = "?";
        } else {
            spec.ty = self.word();
            let ty_span_end = self.cur.peek().map(|(pos, _)| *pos);
            if !spec.ty.is_empty() {
                spec.ty_span = ty_span_start
                    .and_then(|s| ty_span_end.map(|e| (s, e)))
                    .map(|(start, end)| self.to_span_index(start).to(self.to_span_index(end)));
            }
        }
        spec
    }

    /// Parses a `Count` parameter at the current position. This does not check
    /// for 'CountIsNextParam' because that is only used in precision, not
    /// width.
    fn count(&mut self, start: usize) -> (Count, Option<InnerSpan>) {
        if let Some(i) = self.integer() {
            if let Some(end) = self.consume_pos('$') {
                let span = self.to_span_index(start).to(self.to_span_index(end + 1));
                (CountIsParam(i), Some(span))
            } else {
                (CountIs(i), None)
            }
        } else {
            let tmp = self.cur.clone();
            let word = self.word();
            if word.is_empty() {
                self.cur = tmp;
                (CountImplied, None)
            } else if self.consume('$') {
                (CountIsName(Symbol::intern(word)), None)
            } else {
                self.cur = tmp;
                (CountImplied, None)
            }
        }
    }

    /// Parses a word starting at the current position. A word is the same as
    /// Rust identifier, except that it can't start with `_` character.
    fn word(&mut self) -> &'a str {
        let start = match self.cur.peek() {
            Some(&(pos, c)) if rustc_lexer::is_id_start(c) => {
                self.cur.next();
                pos
            }
            _ => {
                return "";
            }
        };
        let mut end = None;
        while let Some(&(pos, c)) = self.cur.peek() {
            if rustc_lexer::is_id_continue(c) {
                self.cur.next();
            } else {
                end = Some(pos);
                break;
            }
        }
        let end = end.unwrap_or(self.input.len());
        let word = &self.input[start..end];
        if word == "_" {
            self.err_with_note(
                "invalid argument name `_`",
                "invalid argument name",
                "argument name cannot be a single underscore",
                self.to_span_index(start).to(self.to_span_index(end)),
            );
        }
        word
    }

    /// Optionally parses an integer at the current position. This doesn't deal
    /// with overflow at all, it's just accumulating digits.
    fn integer(&mut self) -> Option<usize> {
        let mut cur = 0;
        let mut found = false;
        while let Some(&(_, c)) = self.cur.peek() {
            if let Some(i) = c.to_digit(10) {
                cur = cur * 10 + i as usize;
                found = true;
                self.cur.next();
            } else {
                break;
            }
        }
        found.then_some(cur)
    }
}

#[cfg(test)]
mod tests;
