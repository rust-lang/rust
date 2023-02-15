//! Macro support for format strings
//!
//! These structures are used when parsing format strings for the compiler.
//! Parsing does not happen at runtime: structures of `std::fmt::rt` are
//! generated instead.

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(deny(warnings)))
)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
// We want to be able to build this crate with a stable compiler, so no
// `#![feature]` attributes should be added.

pub use Alignment::*;
pub use Count::*;
pub use Piece::*;
pub use Position::*;

use std::iter;
use std::str;
use std::string;

// Note: copied from rustc_span
/// Range inside of a `Span` used for diagnostics when we only have access to relative positions.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct InnerSpan {
    pub start: usize,
    pub end: usize,
}

impl InnerSpan {
    pub fn new(start: usize, end: usize) -> InnerSpan {
        InnerSpan { start, end }
    }
}

/// The location and before/after width of a character whose width has changed from its source code
/// representation
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct InnerWidthMapping {
    /// Index of the character in the source
    pub position: usize,
    /// The inner width in characters
    pub before: usize,
    /// The transformed width in characters
    pub after: usize,
}

impl InnerWidthMapping {
    pub fn new(position: usize, before: usize, after: usize) -> InnerWidthMapping {
        InnerWidthMapping { position, before, after }
    }
}

/// Whether the input string is a literal. If yes, it contains the inner width mappings.
#[derive(Clone, PartialEq, Eq)]
enum InputStringKind {
    NotALiteral,
    Literal { width_mappings: Vec<InnerWidthMapping> },
}

/// The type of format string that we are parsing.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParseMode {
    /// A normal format string as per `format_args!`.
    Format,
    /// An inline assembly template string for `asm!`.
    InlineAsm,
}

#[derive(Copy, Clone)]
struct InnerOffset(usize);

impl InnerOffset {
    fn to(self, end: InnerOffset) -> InnerSpan {
        InnerSpan::new(self.0, end.0)
    }
}

/// A piece is a portion of the format string which represents the next part
/// to emit. These are emitted as a stream by the `Parser` class.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece<'a> {
    /// A literal string which should directly be emitted
    String(&'a str),
    /// This describes that formatting should process the next argument (as
    /// specified inside) for emission.
    NextArgument(Box<Argument<'a>>),
}

/// Representation of an argument specification.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Argument<'a> {
    /// Where to find this argument
    pub position: Position<'a>,
    /// The span of the position indicator. Includes any whitespace in implicit
    /// positions (`{  }`).
    pub position_span: InnerSpan,
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
    /// The `+` or `-` flag.
    pub sign: Option<Sign>,
    /// The `#` flag.
    pub alternate: bool,
    /// The `0` flag.
    pub zero_pad: bool,
    /// The `x` or `X` flag. (Only for `Debug`.)
    pub debug_hex: Option<DebugHex>,
    /// The integer precision to use.
    pub precision: Count<'a>,
    /// The span of the precision formatting flag (for diagnostics).
    pub precision_span: Option<InnerSpan>,
    /// The string width requested for the resulting format.
    pub width: Count<'a>,
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
pub enum Position<'a> {
    /// The argument is implied to be located at an index
    ArgumentImplicitlyIs(usize),
    /// The argument is located at a specific index given in the format,
    ArgumentIs(usize),
    /// The argument has a name.
    ArgumentNamed(&'a str),
}

impl Position<'_> {
    pub fn index(&self) -> Option<usize> {
        match self {
            ArgumentIs(i, ..) | ArgumentImplicitlyIs(i) => Some(*i),
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

/// Enum for the sign flags.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Sign {
    /// The `+` flag.
    Plus,
    /// The `-` flag.
    Minus,
}

/// Enum for the debug hex flags.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DebugHex {
    /// The `x` flag in `{:x?}`.
    Lower,
    /// The `X` flag in `{:X?}`.
    Upper,
}

/// A count is used for the precision and width parameters of an integer, and
/// can reference either an argument or a literal integer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Count<'a> {
    /// The count is specified explicitly.
    CountIs(usize),
    /// The count is specified by the argument with the given name.
    CountIsName(&'a str, InnerSpan),
    /// The count is specified by the argument at the given index.
    CountIsParam(usize),
    /// The count is specified by a star (like in `{:.*}`) that refers to the argument at the given index.
    CountIsStar(usize),
    /// The count is implied and cannot be explicitly specified.
    CountImplied,
}

pub struct ParseError {
    pub description: string::String,
    pub note: Option<string::String>,
    pub label: string::String,
    pub span: InnerSpan,
    pub secondary_label: Option<(string::String, InnerSpan)>,
    pub should_be_replaced_with_positional_argument: bool,
}

/// The parser structure for interpreting the input format string. This is
/// modeled as an iterator over `Piece` structures to form a stream of tokens
/// being output.
///
/// This is a recursive-descent parser for the sake of simplicity, and if
/// necessary there's probably lots of room for improvement performance-wise.
pub struct Parser<'a> {
    mode: ParseMode,
    input: &'a str,
    cur: iter::Peekable<str::CharIndices<'a>>,
    /// Error messages accumulated during parsing
    pub errors: Vec<ParseError>,
    /// Current position of implicit positional argument pointer
    pub curarg: usize,
    /// `Some(raw count)` when the string is "raw", used to position spans correctly
    style: Option<usize>,
    /// Start and end byte offset of every successfully parsed argument
    pub arg_places: Vec<InnerSpan>,
    /// Characters whose length has been changed from their in-code representation
    width_map: Vec<InnerWidthMapping>,
    /// Span of the last opening brace seen, used for error reporting
    last_opening_brace: Option<InnerSpan>,
    /// Whether the source string is comes from `println!` as opposed to `format!` or `print!`
    append_newline: bool,
    /// Whether this formatting string is a literal or it comes from a macro.
    pub is_literal: bool,
    /// Start position of the current line.
    cur_line_start: usize,
    /// Start and end byte offset of every line of the format string. Excludes
    /// newline characters and leading whitespace.
    pub line_spans: Vec<InnerSpan>,
}

impl<'a> Iterator for Parser<'a> {
    type Item = Piece<'a>;

    fn next(&mut self) -> Option<Piece<'a>> {
        if let Some(&(pos, c)) = self.cur.peek() {
            match c {
                '{' => {
                    let curr_last_brace = self.last_opening_brace;
                    let byte_pos = self.to_span_index(pos);
                    let lbrace_end = InnerOffset(byte_pos.0 + self.to_span_width(pos));
                    self.last_opening_brace = Some(byte_pos.to(lbrace_end));
                    self.cur.next();
                    if self.consume('{') {
                        self.last_opening_brace = curr_last_brace;

                        Some(String(self.string(pos + 1)))
                    } else {
                        let arg = self.argument(lbrace_end);
                        if let Some(rbrace_pos) = self.must_consume('}') {
                            if self.is_literal {
                                let lbrace_byte_pos = self.to_span_index(pos);
                                let rbrace_byte_pos = self.to_span_index(rbrace_pos);

                                let width = self.to_span_width(rbrace_pos);

                                self.arg_places.push(
                                    lbrace_byte_pos.to(InnerOffset(rbrace_byte_pos.0 + width)),
                                );
                            }
                        } else {
                            if let Some(&(_, maybe)) = self.cur.peek() {
                                if maybe == '?' {
                                    self.suggest_format();
                                } else {
                                    self.suggest_positional_arg_instead_of_captured_arg(arg);
                                }
                            }
                        }
                        Some(NextArgument(Box::new(arg)))
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
                _ => Some(String(self.string(pos))),
            }
        } else {
            if self.is_literal {
                let span = self.span(self.cur_line_start, self.input.len());
                if self.line_spans.last() != Some(&span) {
                    self.line_spans.push(span);
                }
            }
            None
        }
    }
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given format string
    pub fn new(
        s: &'a str,
        style: Option<usize>,
        snippet: Option<string::String>,
        append_newline: bool,
        mode: ParseMode,
    ) -> Parser<'a> {
        let input_string_kind = find_width_map_from_snippet(snippet, style);
        let (width_map, is_literal) = match input_string_kind {
            InputStringKind::Literal { width_mappings } => (width_mappings, true),
            InputStringKind::NotALiteral => (Vec::new(), false),
        };

        Parser {
            mode,
            input: s,
            cur: s.char_indices().peekable(),
            errors: vec![],
            curarg: 0,
            style,
            arg_places: vec![],
            width_map,
            last_opening_brace: None,
            append_newline,
            is_literal,
            cur_line_start: 0,
            line_spans: vec![],
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
            should_be_replaced_with_positional_argument: false,
        });
    }

    /// Notifies of an error. The message doesn't actually need to be of type
    /// String, but I think it does when this eventually uses conditions so it
    /// might as well start using it now.
    fn err_with_note<
        S1: Into<string::String>,
        S2: Into<string::String>,
        S3: Into<string::String>,
    >(
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
            should_be_replaced_with_positional_argument: false,
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

    fn remap_pos(&self, mut pos: usize) -> InnerOffset {
        for width in &self.width_map {
            if pos > width.position {
                pos += width.before - width.after;
            } else if pos == width.position && width.after == 0 {
                pos += width.before;
            } else {
                break;
            }
        }

        InnerOffset(pos)
    }

    fn to_span_index(&self, pos: usize) -> InnerOffset {
        // This handles the raw string case, the raw argument is the number of #
        // in r###"..."### (we need to add one because of the `r`).
        let raw = self.style.map_or(0, |raw| raw + 1);
        let pos = self.remap_pos(pos);
        InnerOffset(raw + pos.0 + 1)
    }

    fn to_span_width(&self, pos: usize) -> usize {
        let pos = self.remap_pos(pos);
        match self.width_map.iter().find(|w| w.position == pos.0) {
            Some(w) => w.before,
            None => 1,
        }
    }

    fn span(&self, start_pos: usize, end_pos: usize) -> InnerSpan {
        let start = self.to_span_index(start_pos);
        let end = self.to_span_index(end_pos);
        start.to(end)
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
                let description = format!("expected `'}}'`, found `{maybe:?}`");
                let label = "expected `}`".to_owned();
                let (note, secondary_label) = if c == '}' {
                    (
                        Some(
                            "if you intended to print `{`, you can escape it using `{{`".to_owned(),
                        ),
                        self.last_opening_brace
                            .map(|sp| ("because of this opening brace".to_owned(), sp)),
                    )
                } else {
                    (None, None)
                };
                self.errors.push(ParseError {
                    description,
                    note,
                    label,
                    span: pos.to(pos),
                    secondary_label,
                    should_be_replaced_with_positional_argument: false,
                });
                None
            }
        } else {
            let description = format!("expected `{c:?}` but string was terminated");
            // point at closing `"`
            let pos = self.input.len() - if self.append_newline { 1 } else { 0 };
            let pos = self.to_span_index(pos);
            if c == '}' {
                let label = format!("expected `{c:?}`");
                let (note, secondary_label) = if c == '}' {
                    (
                        Some(
                            "if you intended to print `{`, you can escape it using `{{`".to_owned(),
                        ),
                        self.last_opening_brace
                            .map(|sp| ("because of this opening brace".to_owned(), sp)),
                    )
                } else {
                    (None, None)
                };
                self.errors.push(ParseError {
                    description,
                    note,
                    label,
                    span: pos.to(pos),
                    secondary_label,
                    should_be_replaced_with_positional_argument: false,
                });
            } else {
                self.err(description, format!("expected `{c:?}`"), pos.to(pos));
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
                '\n' if self.is_literal => {
                    self.line_spans.push(self.span(self.cur_line_start, pos));
                    self.cur_line_start = pos + 1;
                    self.cur.next();
                }
                _ => {
                    if self.is_literal && pos == self.cur_line_start && c.is_whitespace() {
                        self.cur_line_start = pos + c.len_utf8();
                    }
                    self.cur.next();
                }
            }
        }
        &self.input[start..self.input.len()]
    }

    /// Parses an `Argument` structure, or what's contained within braces inside the format string.
    fn argument(&mut self, start: InnerOffset) -> Argument<'a> {
        let pos = self.position();

        let end = self
            .cur
            .clone()
            .find(|(_, ch)| !ch.is_whitespace())
            .map_or(start, |(end, _)| self.to_span_index(end));
        let position_span = start.to(end);

        let format = match self.mode {
            ParseMode::Format => self.format(),
            ParseMode::InlineAsm => self.inline_asm(),
        };

        // Resolve position after parsing format spec.
        let pos = match pos {
            Some(position) => position,
            None => {
                let i = self.curarg;
                self.curarg += 1;
                ArgumentImplicitlyIs(i)
            }
        };

        Argument { position: pos, position_span, format }
    }

    /// Parses a positional argument for a format. This could either be an
    /// integer index of an argument, a named argument, or a blank string.
    /// Returns `Some(parsed_position)` if the position is not implicitly
    /// consuming a macro argument, `None` if it's the case.
    fn position(&mut self) -> Option<Position<'a>> {
        if let Some(i) = self.integer() {
            Some(ArgumentIs(i))
        } else {
            match self.cur.peek() {
                Some(&(_, c)) if rustc_lexer::is_id_start(c) => Some(ArgumentNamed(self.word())),

                // This is an `ArgumentNext`.
                // Record the fact and do the resolution after parsing the
                // format spec, to make things like `{:.*}` work.
                _ => None,
            }
        }
    }

    fn current_pos(&mut self) -> usize {
        if let Some(&(pos, _)) = self.cur.peek() { pos } else { self.input.len() }
    }

    /// Parses a format specifier at the current position, returning all of the
    /// relevant information in the `FormatSpec` struct.
    fn format(&mut self) -> FormatSpec<'a> {
        let mut spec = FormatSpec {
            fill: None,
            align: AlignUnknown,
            sign: None,
            alternate: false,
            zero_pad: false,
            debug_hex: None,
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
            if let Some((_, '>' | '<' | '^')) = self.cur.clone().nth(1) {
                spec.fill = Some(c);
                self.cur.next();
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
            spec.sign = Some(Sign::Plus);
        } else if self.consume('-') {
            spec.sign = Some(Sign::Minus);
        }
        // Alternate marker
        if self.consume('#') {
            spec.alternate = true;
        }
        // Width and precision
        let mut havewidth = false;

        if self.consume('0') {
            // small ambiguity with '0$' as a format string. In theory this is a
            // '0' flag and then an ill-formatted format string with just a '$'
            // and no count, but this is better if we instead interpret this as
            // no '0' flag and '0$' as the width instead.
            if let Some(end) = self.consume_pos('$') {
                spec.width = CountIsParam(0);
                spec.width_span = Some(self.span(end - 1, end + 1));
                havewidth = true;
            } else {
                spec.zero_pad = true;
            }
        }

        if !havewidth {
            let start = self.current_pos();
            spec.width = self.count(start);
            if spec.width != CountImplied {
                let end = self.current_pos();
                spec.width_span = Some(self.span(start, end));
            }
        }

        if let Some(start) = self.consume_pos('.') {
            if self.consume('*') {
                // Resolve `CountIsNextParam`.
                // We can do this immediately as `position` is resolved later.
                let i = self.curarg;
                self.curarg += 1;
                spec.precision = CountIsStar(i);
            } else {
                spec.precision = self.count(start + 1);
            }
            let end = self.current_pos();
            spec.precision_span = Some(self.span(start, end));
        }

        let ty_span_start = self.current_pos();
        // Optional radix followed by the actual format specifier
        if self.consume('x') {
            if self.consume('?') {
                spec.debug_hex = Some(DebugHex::Lower);
                spec.ty = "?";
            } else {
                spec.ty = "x";
            }
        } else if self.consume('X') {
            if self.consume('?') {
                spec.debug_hex = Some(DebugHex::Upper);
                spec.ty = "?";
            } else {
                spec.ty = "X";
            }
        } else if self.consume('?') {
            spec.ty = "?";
        } else {
            spec.ty = self.word();
            if !spec.ty.is_empty() {
                let ty_span_end = self.current_pos();
                spec.ty_span = Some(self.span(ty_span_start, ty_span_end));
            }
        }
        spec
    }

    /// Parses an inline assembly template modifier at the current position, returning the modifier
    /// in the `ty` field of the `FormatSpec` struct.
    fn inline_asm(&mut self) -> FormatSpec<'a> {
        let mut spec = FormatSpec {
            fill: None,
            align: AlignUnknown,
            sign: None,
            alternate: false,
            zero_pad: false,
            debug_hex: None,
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

        let ty_span_start = self.current_pos();
        spec.ty = self.word();
        if !spec.ty.is_empty() {
            let ty_span_end = self.current_pos();
            spec.ty_span = Some(self.span(ty_span_start, ty_span_end));
        }

        spec
    }

    /// Parses a `Count` parameter at the current position. This does not check
    /// for 'CountIsNextParam' because that is only used in precision, not
    /// width.
    fn count(&mut self, start: usize) -> Count<'a> {
        if let Some(i) = self.integer() {
            if self.consume('$') { CountIsParam(i) } else { CountIs(i) }
        } else {
            let tmp = self.cur.clone();
            let word = self.word();
            if word.is_empty() {
                self.cur = tmp;
                CountImplied
            } else if let Some(end) = self.consume_pos('$') {
                let name_span = self.span(start, end);
                CountIsName(word, name_span)
            } else {
                self.cur = tmp;
                CountImplied
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
                self.span(start, end),
            );
        }
        word
    }

    fn integer(&mut self) -> Option<usize> {
        let mut cur: usize = 0;
        let mut found = false;
        let mut overflow = false;
        let start = self.current_pos();
        while let Some(&(_, c)) = self.cur.peek() {
            if let Some(i) = c.to_digit(10) {
                let (tmp, mul_overflow) = cur.overflowing_mul(10);
                let (tmp, add_overflow) = tmp.overflowing_add(i as usize);
                if mul_overflow || add_overflow {
                    overflow = true;
                }
                cur = tmp;
                found = true;
                self.cur.next();
            } else {
                break;
            }
        }

        if overflow {
            let end = self.current_pos();
            let overflowed_int = &self.input[start..end];
            self.err(
                format!(
                    "integer `{}` does not fit into the type `usize` whose range is `0..={}`",
                    overflowed_int,
                    usize::MAX
                ),
                "integer out of range for `usize`",
                self.span(start, end),
            );
        }

        found.then_some(cur)
    }

    fn suggest_format(&mut self) {
        if let (Some(pos), Some(_)) = (self.consume_pos('?'), self.consume_pos(':')) {
            let word = self.word();
            let _end = self.current_pos();
            let pos = self.to_span_index(pos);
            self.errors.insert(
                0,
                ParseError {
                    description: "expected format parameter to occur after `:`".to_owned(),
                    note: Some(format!("`?` comes after `:`, try `{}:{}` instead", word, "?")),
                    label: "expected `?` to occur after `:`".to_owned(),
                    span: pos.to(pos),
                    secondary_label: None,
                    should_be_replaced_with_positional_argument: false,
                },
            );
        }
    }

    fn suggest_positional_arg_instead_of_captured_arg(&mut self, arg: Argument<'a>) {
        if let Some(end) = self.consume_pos('.') {
            let byte_pos = self.to_span_index(end);
            let start = InnerOffset(byte_pos.0 + 1);
            let field = self.argument(start);
            // We can only parse `foo.bar` field access, any deeper nesting,
            // or another type of expression, like method calls, are not supported
            if !self.consume('}') {
                return;
            }
            if let ArgumentNamed(_) = arg.position {
                if let ArgumentNamed(_) = field.position {
                    self.errors.insert(
                        0,
                        ParseError {
                            description: "field access isn't supported".to_string(),
                            note: None,
                            label: "not supported".to_string(),
                            span: InnerSpan::new(arg.position_span.start, field.position_span.end),
                            secondary_label: None,
                            should_be_replaced_with_positional_argument: true,
                        },
                    );
                }
            }
        }
    }
}

/// Finds the indices of all characters that have been processed and differ between the actual
/// written code (code snippet) and the `InternedString` that gets processed in the `Parser`
/// in order to properly synthesise the intra-string `Span`s for error diagnostics.
fn find_width_map_from_snippet(
    snippet: Option<string::String>,
    str_style: Option<usize>,
) -> InputStringKind {
    let snippet = match snippet {
        Some(ref s) if s.starts_with('"') || s.starts_with("r\"") || s.starts_with("r#") => s,
        _ => return InputStringKind::NotALiteral,
    };

    if str_style.is_some() {
        return InputStringKind::Literal { width_mappings: Vec::new() };
    }

    let snippet = &snippet[1..snippet.len() - 1];

    let mut s = snippet.char_indices();
    let mut width_mappings = vec![];
    while let Some((pos, c)) = s.next() {
        match (c, s.clone().next()) {
            // skip whitespace and empty lines ending in '\\'
            ('\\', Some((_, '\n'))) => {
                let _ = s.next();
                let mut width = 2;

                while let Some((_, c)) = s.clone().next() {
                    if matches!(c, ' ' | '\n' | '\t') {
                        width += 1;
                        let _ = s.next();
                    } else {
                        break;
                    }
                }

                width_mappings.push(InnerWidthMapping::new(pos, width, 0));
            }
            ('\\', Some((_, 'n' | 't' | 'r' | '0' | '\\' | '\'' | '\"'))) => {
                width_mappings.push(InnerWidthMapping::new(pos, 2, 1));
                let _ = s.next();
            }
            ('\\', Some((_, 'x'))) => {
                // consume `\xAB` literal
                s.nth(2);
                width_mappings.push(InnerWidthMapping::new(pos, 4, 1));
            }
            ('\\', Some((_, 'u'))) => {
                let mut width = 2;
                let _ = s.next();

                if let Some((_, next_c)) = s.next() {
                    if next_c == '{' {
                        // consume up to 6 hexanumeric chars
                        let digits_len =
                            s.clone().take(6).take_while(|(_, c)| c.is_digit(16)).count();

                        let len_utf8 = s
                            .as_str()
                            .get(..digits_len)
                            .and_then(|digits| u32::from_str_radix(digits, 16).ok())
                            .and_then(char::from_u32)
                            .map_or(1, char::len_utf8);

                        // Skip the digits, for chars that encode to more than 1 utf-8 byte
                        // exclude as many digits as it is greater than 1 byte
                        //
                        // So for a 3 byte character, exclude 2 digits
                        let required_skips = digits_len.saturating_sub(len_utf8.saturating_sub(1));

                        // skip '{' and '}' also
                        width += required_skips + 2;

                        s.nth(digits_len);
                    } else if next_c.is_digit(16) {
                        width += 1;

                        // We suggest adding `{` and `}` when appropriate, accept it here as if
                        // it were correct
                        let mut i = 0; // consume up to 6 hexanumeric chars
                        while let (Some((_, c)), _) = (s.next(), i < 6) {
                            if c.is_digit(16) {
                                width += 1;
                            } else {
                                break;
                            }
                            i += 1;
                        }
                    }
                }

                width_mappings.push(InnerWidthMapping::new(pos, width, 1));
            }
            _ => {}
        }
    }

    InputStringKind::Literal { width_mappings }
}

// Assert a reasonable size for `Piece`
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(Piece<'_>, 16);

#[cfg(test)]
mod tests;
