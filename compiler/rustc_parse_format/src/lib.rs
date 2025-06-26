//! Macro support for format strings
//!
//! These structures are used when parsing format strings for the compiler.
//! Parsing does not happen at runtime: structures of `std::fmt::rt` are
//! generated instead.

// tidy-alphabetical-start
// We want to be able to build this crate with a stable compiler,
// so no `#![feature]` attributes should be added.
#![deny(unstable_features)]
#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(deny(warnings)))
)]
// tidy-alphabetical-end

use std::ops::Range;

pub use Alignment::*;
pub use Count::*;
pub use Position::*;

/// The type of format string that we are parsing.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParseMode {
    /// A normal format string as per `format_args!`.
    Format,
    /// An inline assembly template string for `asm!`.
    InlineAsm,
    /// A format string for use in diagnostic attributes.
    ///
    /// Similar to `format_args!`, however only named ("captured") arguments
    /// are allowed, and no format modifiers are permitted.
    Diagnostic,
}

/// A piece is a portion of the format string which represents the next part
/// to emit. These are emitted as a stream by the `Parser` class.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece<'input> {
    /// A literal string which should directly be emitted
    Lit(&'input str),
    /// This describes that formatting should process the next argument (as
    /// specified inside) for emission.
    NextArgument(Box<Argument<'input>>),
}

/// Representation of an argument specification.
#[derive(Clone, Debug, PartialEq)]
pub struct Argument<'input> {
    /// Where to find this argument
    pub position: Position<'input>,
    /// The span of the position indicator. Includes any whitespace in implicit
    /// positions (`{  }`).
    pub position_span: Range<usize>,
    /// How to format the argument
    pub format: FormatSpec<'input>,
}

impl<'input> Argument<'input> {
    pub fn is_identifier(&self) -> bool {
        matches!(self.position, Position::ArgumentNamed(_)) && self.format == FormatSpec::default()
    }
}

/// Specification for the formatting of an argument in the format string.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FormatSpec<'input> {
    /// Optionally specified character to fill alignment with.
    pub fill: Option<char>,
    /// Span of the optionally specified fill character.
    pub fill_span: Option<Range<usize>>,
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
    pub precision: Count<'input>,
    /// The span of the precision formatting flag (for diagnostics).
    pub precision_span: Option<Range<usize>>,
    /// The string width requested for the resulting format.
    pub width: Count<'input>,
    /// The span of the width formatting flag (for diagnostics).
    pub width_span: Option<Range<usize>>,
    /// The descriptor string representing the name of the format desired for
    /// this argument, this can be empty or any number of characters, although
    /// it is required to be one word.
    pub ty: &'input str,
    /// The span of the descriptor string (for diagnostics).
    pub ty_span: Option<Range<usize>>,
}

/// Enum describing where an argument for a format can be located.
#[derive(Clone, Debug, PartialEq)]
pub enum Position<'input> {
    /// The argument is implied to be located at an index
    ArgumentImplicitlyIs(usize),
    /// The argument is located at a specific index given in the format,
    ArgumentIs(usize),
    /// The argument has a name.
    ArgumentNamed(&'input str),
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
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum Alignment {
    /// The value will be aligned to the left.
    AlignLeft,
    /// The value will be aligned to the right.
    AlignRight,
    /// The value will be aligned in the center.
    AlignCenter,
    /// The value will take on a default alignment.
    #[default]
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
#[derive(Clone, Debug, PartialEq, Default)]
pub enum Count<'input> {
    /// The count is specified explicitly.
    CountIs(u16),
    /// The count is specified by the argument with the given name.
    CountIsName(&'input str, Range<usize>),
    /// The count is specified by the argument at the given index.
    CountIsParam(usize),
    /// The count is specified by a star (like in `{:.*}`) that refers to the argument at the given index.
    CountIsStar(usize),
    /// The count is implied and cannot be explicitly specified.
    #[default]
    CountImplied,
}

pub struct ParseError {
    pub description: String,
    pub note: Option<String>,
    pub label: String,
    pub span: Range<usize>,
    pub secondary_label: Option<(String, Range<usize>)>,
    pub suggestion: Suggestion,
}

pub enum Suggestion {
    None,
    /// Replace inline argument with positional argument:
    /// `format!("{foo.bar}")` -> `format!("{}", foo.bar)`
    UsePositional,
    /// Remove `r#` from identifier:
    /// `format!("{r#foo}")` -> `format!("{foo}")`
    RemoveRawIdent(Range<usize>),
    /// Reorder format parameter:
    /// `format!("{foo:?#}")` -> `format!("{foo:#?}")`
    /// `format!("{foo:?x}")` -> `format!("{foo:x?}")`
    /// `format!("{foo:?X}")` -> `format!("{foo:X?}")`
    ReorderFormatParameter(Range<usize>, String),
}

/// The parser structure for interpreting the input format string. This is
/// modeled as an iterator over `Piece` structures to form a stream of tokens
/// being output.
///
/// This is a recursive-descent parser for the sake of simplicity, and if
/// necessary there's probably lots of room for improvement performance-wise.
pub struct Parser<'input> {
    mode: ParseMode,
    /// Input to be parsed
    input: &'input str,
    /// Tuples of the span in the code snippet (input as written before being unescaped), the pos in input, and the char in input
    input_vec: Vec<(Range<usize>, usize, char)>,
    /// Index into input_vec
    input_vec_index: usize,
    /// Error messages accumulated during parsing
    pub errors: Vec<ParseError>,
    /// Current position of implicit positional argument pointer
    pub curarg: usize,
    /// Start and end byte offset of every successfully parsed argument
    pub arg_places: Vec<Range<usize>>,
    /// Span of the last opening brace seen, used for error reporting
    last_open_brace: Option<Range<usize>>,
    /// Whether this formatting string was written directly in the source. This controls whether we
    /// can use spans to refer into it and give better error messages.
    /// N.B: This does _not_ control whether implicit argument captures can be used.
    pub is_source_literal: bool,
    /// Index to the end of the literal snippet
    end_of_snippet: usize,
    /// Start position of the current line.
    cur_line_start: usize,
    /// Start and end byte offset of every line of the format string. Excludes
    /// newline characters and leading whitespace.
    pub line_spans: Vec<Range<usize>>,
}

impl<'input> Iterator for Parser<'input> {
    type Item = Piece<'input>;

    fn next(&mut self) -> Option<Piece<'input>> {
        if let Some((Range { start, end }, idx, ch)) = self.peek() {
            match ch {
                '{' => {
                    self.input_vec_index += 1;
                    if let Some((_, i, '{')) = self.peek() {
                        self.input_vec_index += 1;
                        // double open brace escape: "{{"
                        // next state after this is either end-of-input or seen-a-brace
                        Some(Piece::Lit(self.string(i)))
                    } else {
                        // single open brace
                        self.last_open_brace = Some(start..end);
                        let arg = self.argument();
                        self.ws();
                        if let Some((close_brace_range, _)) = self.consume_pos('}') {
                            if self.is_source_literal {
                                self.arg_places.push(start..close_brace_range.end);
                            }
                        } else {
                            self.missing_closing_brace(&arg);
                        }

                        Some(Piece::NextArgument(Box::new(arg)))
                    }
                }
                '}' => {
                    self.input_vec_index += 1;
                    if let Some((_, i, '}')) = self.peek() {
                        self.input_vec_index += 1;
                        // double close brace escape: "}}"
                        // next state after this is either end-of-input or start
                        Some(Piece::Lit(self.string(i)))
                    } else {
                        // error: single close brace without corresponding open brace
                        self.errors.push(ParseError {
                            description: "unmatched `}` found".into(),
                            note: Some(
                                "if you intended to print `}`, you can escape it using `}}`".into(),
                            ),
                            label: "unmatched `}`".into(),
                            span: start..end,
                            secondary_label: None,
                            suggestion: Suggestion::None,
                        });
                        None
                    }
                }
                _ => Some(Piece::Lit(self.string(idx))),
            }
        } else {
            // end of input
            if self.is_source_literal {
                let span = self.cur_line_start..self.end_of_snippet;
                if self.line_spans.last() != Some(&span) {
                    self.line_spans.push(span);
                }
            }
            None
        }
    }
}

impl<'input> Parser<'input> {
    /// Creates a new parser for the given unescaped input string and
    /// optional code snippet (the input as written before being unescaped),
    /// where `style` is `Some(nr_hashes)` when the snippet is a raw string with that many hashes.
    /// If the input comes via `println` or `panic`, then it has a newline already appended,
    /// which is reflected in the `appended_newline` parameter.
    pub fn new(
        input: &'input str,
        style: Option<usize>,
        snippet: Option<String>,
        appended_newline: bool,
        mode: ParseMode,
    ) -> Self {
        let quote_offset = style.map_or(1, |nr_hashes| nr_hashes + 2);

        let (is_source_literal, end_of_snippet, pre_input_vec) = if let Some(snippet) = snippet {
            if let Some(nr_hashes) = style {
                // snippet is a raw string, which starts with 'r', a number of hashes, and a quote
                // and ends with a quote and the same number of hashes
                (true, snippet.len() - nr_hashes - 1, vec![])
            } else {
                // snippet is not a raw string
                if snippet.starts_with('"') {
                    // snippet looks like an ordinary string literal
                    // check whether it is the escaped version of input
                    let without_quotes = &snippet[1..snippet.len() - 1];
                    let (mut ok, mut vec) = (true, vec![]);
                    let mut chars = input.chars();
                    rustc_literal_escaper::unescape_str(without_quotes, |range, res| match res {
                        Ok(ch) if ok && chars.next().is_some_and(|c| ch == c) => {
                            vec.push((range, ch));
                        }
                        _ => {
                            ok = false;
                            vec = vec![];
                        }
                    });
                    let end = vec.last().map(|(r, _)| r.end).unwrap_or(0);
                    if ok {
                        if appended_newline {
                            if chars.as_str() == "\n" {
                                vec.push((end..end + 1, '\n'));
                                (true, 1 + end, vec)
                            } else {
                                (false, snippet.len(), vec![])
                            }
                        } else if chars.as_str() == "" {
                            (true, 1 + end, vec)
                        } else {
                            (false, snippet.len(), vec![])
                        }
                    } else {
                        (false, snippet.len(), vec![])
                    }
                } else {
                    // snippet is not a raw string and does not start with '"'
                    (false, snippet.len(), vec![])
                }
            }
        } else {
            // snippet is None
            (false, input.len() - if appended_newline { 1 } else { 0 }, vec![])
        };

        let input_vec: Vec<(Range<usize>, usize, char)> = if pre_input_vec.is_empty() {
            // Snippet is *not* input before unescaping, so spans pointing at it will be incorrect.
            // This can happen with proc macros that respan generated literals.
            input
                .char_indices()
                .map(|(idx, c)| {
                    let i = idx + quote_offset;
                    (i..i + c.len_utf8(), idx, c)
                })
                .collect()
        } else {
            // Snippet is input before unescaping
            input
                .char_indices()
                .zip(pre_input_vec)
                .map(|((i, c), (r, _))| (r.start + quote_offset..r.end + quote_offset, i, c))
                .collect()
        };

        Parser {
            mode,
            input,
            input_vec,
            input_vec_index: 0,
            errors: vec![],
            curarg: 0,
            arg_places: vec![],
            last_open_brace: None,
            is_source_literal,
            end_of_snippet,
            cur_line_start: quote_offset,
            line_spans: vec![],
        }
    }

    /// Peeks at the current position, without incrementing the pointer.
    pub fn peek(&self) -> Option<(Range<usize>, usize, char)> {
        self.input_vec.get(self.input_vec_index).cloned()
    }

    /// Peeks at the current position + 1, without incrementing the pointer.
    pub fn peek_ahead(&self) -> Option<(Range<usize>, usize, char)> {
        self.input_vec.get(self.input_vec_index + 1).cloned()
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
    fn consume_pos(&mut self, ch: char) -> Option<(Range<usize>, usize)> {
        if let Some((r, i, c)) = self.peek()
            && ch == c
        {
            self.input_vec_index += 1;
            return Some((r, i));
        }

        None
    }

    /// Called if a closing brace was not found.
    fn missing_closing_brace(&mut self, arg: &Argument<'_>) {
        let (range, description) = if let Some((r, _, c)) = self.peek() {
            (r.start..r.start, format!("expected `}}`, found `{}`", c.escape_debug()))
        } else {
            (
                // point at closing `"`
                self.end_of_snippet..self.end_of_snippet,
                "expected `}` but string was terminated".to_owned(),
            )
        };

        let (note, secondary_label) = if arg.format.fill == Some('}') {
            (
                Some("the character `}` is interpreted as a fill character because of the `:` that precedes it".to_owned()),
                arg.format.fill_span.clone().map(|sp| ("this is not interpreted as a formatting closing brace".to_owned(), sp)),
            )
        } else {
            (
                Some("if you intended to print `{`, you can escape it using `{{`".to_owned()),
                self.last_open_brace
                    .clone()
                    .map(|sp| ("because of this opening brace".to_owned(), sp)),
            )
        };

        self.errors.push(ParseError {
            description,
            note,
            label: "expected `}`".to_owned(),
            span: range.start..range.start,
            secondary_label,
            suggestion: Suggestion::None,
        });

        if let Some((_, _, c)) = self.peek() {
            match c {
                '?' => self.suggest_format_debug(),
                '<' | '^' | '>' => self.suggest_format_align(c),
                _ => self.suggest_positional_arg_instead_of_captured_arg(arg),
            }
        }
    }

    /// Consumes all whitespace characters until the first non-whitespace character
    fn ws(&mut self) {
        let rest = &self.input_vec[self.input_vec_index..];
        let step = rest.iter().position(|&(_, _, c)| !c.is_whitespace()).unwrap_or(rest.len());
        self.input_vec_index += step;
    }

    /// Parses all of a string which is to be considered a "raw literal" in a
    /// format string. This is everything outside of the braces.
    fn string(&mut self, start: usize) -> &'input str {
        while let Some((r, i, c)) = self.peek() {
            match c {
                '{' | '}' => {
                    return &self.input[start..i];
                }
                '\n' if self.is_source_literal => {
                    self.input_vec_index += 1;
                    self.line_spans.push(self.cur_line_start..r.start);
                    self.cur_line_start = r.end;
                }
                _ => {
                    self.input_vec_index += 1;
                    if self.is_source_literal && r.start == self.cur_line_start && c.is_whitespace()
                    {
                        self.cur_line_start = r.end;
                    }
                }
            }
        }
        &self.input[start..]
    }

    /// Parses an `Argument` structure, or what's contained within braces inside the format string.
    fn argument(&mut self) -> Argument<'input> {
        let start_idx = self.input_vec_index;

        let position = self.position();
        self.ws();

        let end_idx = self.input_vec_index;

        let format = match self.mode {
            ParseMode::Format => self.format(),
            ParseMode::InlineAsm => self.inline_asm(),
            ParseMode::Diagnostic => self.diagnostic(),
        };

        // Resolve position after parsing format spec.
        let position = position.unwrap_or_else(|| {
            let i = self.curarg;
            self.curarg += 1;
            ArgumentImplicitlyIs(i)
        });

        let position_span =
            self.input_vec_index2range(start_idx).start..self.input_vec_index2range(end_idx).start;
        Argument { position, position_span, format }
    }

    /// Parses a positional argument for a format. This could either be an
    /// integer index of an argument, a named argument, or a blank string.
    /// Returns `Some(parsed_position)` if the position is not implicitly
    /// consuming a macro argument, `None` if it's the case.
    fn position(&mut self) -> Option<Position<'input>> {
        if let Some(i) = self.integer() {
            Some(ArgumentIs(i.into()))
        } else {
            match self.peek() {
                Some((range, _, c)) if rustc_lexer::is_id_start(c) => {
                    let start = range.start;
                    let word = self.word();

                    // Recover from `r#ident` in format strings.
                    if word == "r"
                        && let Some((r, _, '#')) = self.peek()
                        && self.peek_ahead().is_some_and(|(_, _, c)| rustc_lexer::is_id_start(c))
                    {
                        self.input_vec_index += 1;
                        let prefix_end = r.end;
                        let word = self.word();
                        let prefix_span = start..prefix_end;
                        let full_span =
                            start..self.input_vec_index2range(self.input_vec_index).start;
                        self.errors.insert(0, ParseError {
                                    description: "raw identifiers are not supported".to_owned(),
                                    note: Some("identifiers in format strings can be keywords and don't need to be prefixed with `r#`".to_string()),
                                    label: "raw identifier used here".to_owned(),
                                    span: full_span,
                                    secondary_label: None,
                                    suggestion: Suggestion::RemoveRawIdent(prefix_span),
                                });
                        return Some(ArgumentNamed(word));
                    }

                    Some(ArgumentNamed(word))
                }
                // This is an `ArgumentNext`.
                // Record the fact and do the resolution after parsing the
                // format spec, to make things like `{:.*}` work.
                _ => None,
            }
        }
    }

    fn input_vec_index2pos(&self, index: usize) -> usize {
        if let Some((_, pos, _)) = self.input_vec.get(index) { *pos } else { self.input.len() }
    }

    fn input_vec_index2range(&self, index: usize) -> Range<usize> {
        if let Some((r, _, _)) = self.input_vec.get(index) {
            r.clone()
        } else {
            self.end_of_snippet..self.end_of_snippet
        }
    }

    /// Parses a format specifier at the current position, returning all of the
    /// relevant information in the `FormatSpec` struct.
    fn format(&mut self) -> FormatSpec<'input> {
        let mut spec = FormatSpec::default();

        if !self.consume(':') {
            return spec;
        }

        // fill character
        if let (Some((r, _, c)), Some((_, _, '>' | '<' | '^'))) = (self.peek(), self.peek_ahead()) {
            self.input_vec_index += 1;
            spec.fill = Some(c);
            spec.fill_span = Some(r);
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

        if let Some((range, _)) = self.consume_pos('0') {
            // small ambiguity with '0$' as a format string. In theory this is a
            // '0' flag and then an ill-formatted format string with just a '$'
            // and no count, but this is better if we instead interpret this as
            // no '0' flag and '0$' as the width instead.
            if let Some((r, _)) = self.consume_pos('$') {
                spec.width = CountIsParam(0);
                spec.width_span = Some(range.start..r.end);
                havewidth = true;
            } else {
                spec.zero_pad = true;
            }
        }

        if !havewidth {
            let start_idx = self.input_vec_index;
            spec.width = self.count();
            if spec.width != CountImplied {
                let end = self.input_vec_index2range(self.input_vec_index).start;
                spec.width_span = Some(self.input_vec_index2range(start_idx).start..end);
            }
        }

        if let Some((range, _)) = self.consume_pos('.') {
            if self.consume('*') {
                // Resolve `CountIsNextParam`.
                // We can do this immediately as `position` is resolved later.
                let i = self.curarg;
                self.curarg += 1;
                spec.precision = CountIsStar(i);
            } else {
                spec.precision = self.count();
            }
            spec.precision_span =
                Some(range.start..self.input_vec_index2range(self.input_vec_index).start);
        }

        let start_idx = self.input_vec_index;
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
        } else if let Some((range, _)) = self.consume_pos('?') {
            spec.ty = "?";
            if let Some((r, _, c @ ('#' | 'x' | 'X'))) = self.peek() {
                self.errors.insert(
                    0,
                    ParseError {
                        description: format!("expected `}}`, found `{c}`"),
                        note: None,
                        label: "expected `'}'`".into(),
                        span: r.clone(),
                        secondary_label: None,
                        suggestion: Suggestion::ReorderFormatParameter(
                            range.start..r.end,
                            format!("{c}?"),
                        ),
                    },
                );
            }
        } else {
            spec.ty = self.word();
            if !spec.ty.is_empty() {
                let start = self.input_vec_index2range(start_idx).start;
                let end = self.input_vec_index2range(self.input_vec_index).start;
                spec.ty_span = Some(start..end);
            }
        }
        spec
    }

    /// Parses an inline assembly template modifier at the current position, returning the modifier
    /// in the `ty` field of the `FormatSpec` struct.
    fn inline_asm(&mut self) -> FormatSpec<'input> {
        let mut spec = FormatSpec::default();

        if !self.consume(':') {
            return spec;
        }

        let start_idx = self.input_vec_index;
        spec.ty = self.word();
        if !spec.ty.is_empty() {
            let start = self.input_vec_index2range(start_idx).start;
            let end = self.input_vec_index2range(self.input_vec_index).start;
            spec.ty_span = Some(start..end);
        }

        spec
    }

    /// Always returns an empty `FormatSpec`
    fn diagnostic(&mut self) -> FormatSpec<'input> {
        let mut spec = FormatSpec::default();

        let Some((Range { start, .. }, start_idx)) = self.consume_pos(':') else {
            return spec;
        };

        spec.ty = self.string(start_idx);
        spec.ty_span = {
            let end = self.input_vec_index2range(self.input_vec_index).start;
            Some(start..end)
        };
        spec
    }

    /// Parses a `Count` parameter at the current position. This does not check
    /// for 'CountIsNextParam' because that is only used in precision, not
    /// width.
    fn count(&mut self) -> Count<'input> {
        if let Some(i) = self.integer() {
            if self.consume('$') { CountIsParam(i.into()) } else { CountIs(i) }
        } else {
            let start_idx = self.input_vec_index;
            let word = self.word();
            if word.is_empty() {
                CountImplied
            } else if let Some((r, _)) = self.consume_pos('$') {
                CountIsName(word, self.input_vec_index2range(start_idx).start..r.start)
            } else {
                self.input_vec_index = start_idx;
                CountImplied
            }
        }
    }

    /// Parses a word starting at the current position. A word is the same as a
    /// Rust identifier, except that it can't start with `_` character.
    fn word(&mut self) -> &'input str {
        let index = self.input_vec_index;
        match self.peek() {
            Some((ref r, i, c)) if rustc_lexer::is_id_start(c) => {
                self.input_vec_index += 1;
                (r.start, i)
            }
            _ => {
                return "";
            }
        };
        let (err_end, end): (usize, usize) = loop {
            if let Some((ref r, i, c)) = self.peek() {
                if rustc_lexer::is_id_continue(c) {
                    self.input_vec_index += 1;
                } else {
                    break (r.start, i);
                }
            } else {
                break (self.end_of_snippet, self.input.len());
            }
        };

        let word = &self.input[self.input_vec_index2pos(index)..end];
        if word == "_" {
            self.errors.push(ParseError {
                description: "invalid argument name `_`".into(),
                note: Some("argument name cannot be a single underscore".into()),
                label: "invalid argument name".into(),
                span: self.input_vec_index2range(index).start..err_end,
                secondary_label: None,
                suggestion: Suggestion::None,
            });
        }
        word
    }

    fn integer(&mut self) -> Option<u16> {
        let mut cur: u16 = 0;
        let mut found = false;
        let mut overflow = false;
        let start_index = self.input_vec_index;
        while let Some((_, _, c)) = self.peek() {
            if let Some(i) = c.to_digit(10) {
                self.input_vec_index += 1;
                let (tmp, mul_overflow) = cur.overflowing_mul(10);
                let (tmp, add_overflow) = tmp.overflowing_add(i as u16);
                if mul_overflow || add_overflow {
                    overflow = true;
                }
                cur = tmp;
                found = true;
            } else {
                break;
            }
        }

        if overflow {
            let overflowed_int = &self.input[self.input_vec_index2pos(start_index)
                ..self.input_vec_index2pos(self.input_vec_index)];
            self.errors.push(ParseError {
                description: format!(
                    "integer `{}` does not fit into the type `u16` whose range is `0..={}`",
                    overflowed_int,
                    u16::MAX
                ),
                note: None,
                label: "integer out of range for `u16`".into(),
                span: self.input_vec_index2range(start_index).start
                    ..self.input_vec_index2range(self.input_vec_index).end,
                secondary_label: None,
                suggestion: Suggestion::None,
            });
        }

        found.then_some(cur)
    }

    fn suggest_format_debug(&mut self) {
        if let (Some((range, _)), Some(_)) = (self.consume_pos('?'), self.consume_pos(':')) {
            let word = self.word();
            self.errors.insert(
                0,
                ParseError {
                    description: "expected format parameter to occur after `:`".to_owned(),
                    note: Some(format!("`?` comes after `:`, try `{}:{}` instead", word, "?")),
                    label: "expected `?` to occur after `:`".to_owned(),
                    span: range,
                    secondary_label: None,
                    suggestion: Suggestion::None,
                },
            );
        }
    }

    fn suggest_format_align(&mut self, alignment: char) {
        if let Some((range, _)) = self.consume_pos(alignment) {
            self.errors.insert(
                0,
                ParseError {
                    description: "expected format parameter to occur after `:`".to_owned(),
                    note: None,
                    label: format!("expected `{}` to occur after `:`", alignment),
                    span: range,
                    secondary_label: None,
                    suggestion: Suggestion::None,
                },
            );
        }
    }

    fn suggest_positional_arg_instead_of_captured_arg(&mut self, arg: &Argument<'_>) {
        // If the argument is not an identifier, it is not a field access.
        if !arg.is_identifier() {
            return;
        }

        if let Some((_range, _pos)) = self.consume_pos('.') {
            let field = self.argument();
            // We can only parse simple `foo.bar` field access or `foo.0` tuple index access, any
            // deeper nesting, or another type of expression, like method calls, are not supported
            if !self.consume('}') {
                return;
            }
            if let ArgumentNamed(_) = arg.position {
                match field.position {
                    ArgumentNamed(_) => {
                        self.errors.insert(
                            0,
                            ParseError {
                                description: "field access isn't supported".to_string(),
                                note: None,
                                label: "not supported".to_string(),
                                span: arg.position_span.start..field.position_span.end,
                                secondary_label: None,
                                suggestion: Suggestion::UsePositional,
                            },
                        );
                    }
                    ArgumentIs(_) => {
                        self.errors.insert(
                            0,
                            ParseError {
                                description: "tuple index access isn't supported".to_string(),
                                note: None,
                                label: "not supported".to_string(),
                                span: arg.position_span.start..field.position_span.end,
                                secondary_label: None,
                                suggestion: Suggestion::UsePositional,
                            },
                        );
                    }
                    _ => {}
                };
            }
        }
    }
}

// Assert a reasonable size for `Piece`
#[cfg(all(test, target_pointer_width = "64"))]
rustc_index::static_assert_size!(Piece<'_>, 16);

#[cfg(test)]
mod tests;
