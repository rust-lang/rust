// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Macro support for format strings
//!
//! These structures are used when parsing format strings for the compiler.
//! Parsing does not happen at runtime: structures of `std::fmt::rt` are
//! generated instead.

#![crate_name = "fmt_macros"]
#![experimental]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![feature(macro_rules, globs, import_shadowing)]

use std::char;
use std::str;
use std::string;

/// A piece is a portion of the format string which represents the next part
/// to emit. These are emitted as a stream by the `Parser` class.
#[deriving(PartialEq)]
pub enum Piece<'a> {
    /// A literal string which should directly be emitted
    String(&'a str),
    /// This describes that formatting should process the next argument (as
    /// specified inside) for emission.
    NextArgument(Argument<'a>),
}

/// Representation of an argument specification.
#[deriving(PartialEq)]
pub struct Argument<'a> {
    /// Where to find this argument
    pub position: Position<'a>,
    /// How to format the argument
    pub format: FormatSpec<'a>,
}

/// Specification for the formatting of an argument in the format string.
#[deriving(PartialEq)]
pub struct FormatSpec<'a> {
    /// Optionally specified character to fill alignment with
    pub fill: Option<char>,
    /// Optionally specified alignment
    pub align: Alignment,
    /// Packed version of various flags provided
    pub flags: uint,
    /// The integer precision to use
    pub precision: Count<'a>,
    /// The string width requested for the resulting format
    pub width: Count<'a>,
    /// The descriptor string representing the name of the format desired for
    /// this argument, this can be empty or any number of characters, although
    /// it is required to be one word.
    pub ty: &'a str
}

/// Enum describing where an argument for a format can be located.
#[deriving(PartialEq)]
pub enum Position<'a> {
    /// The argument will be in the next position. This is the default.
    ArgumentNext,
    /// The argument is located at a specific index.
    ArgumentIs(uint),
    /// The argument has a name.
    ArgumentNamed(&'a str),
}

/// Enum of alignments which are supported.
#[deriving(PartialEq)]
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
#[deriving(PartialEq)]
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
}

/// A count is used for the precision and width parameters of an integer, and
/// can reference either an argument or a literal integer.
#[deriving(PartialEq)]
pub enum Count<'a> {
    /// The count is specified explicitly.
    CountIs(uint),
    /// The count is specified by the argument with the given name.
    CountIsName(&'a str),
    /// The count is specified by the argument at the given index.
    CountIsParam(uint),
    /// The count is specified by the next parameter.
    CountIsNextParam,
    /// The count is implied and cannot be explicitly specified.
    CountImplied,
}

/// The parser structure for interpreting the input format string. This is
/// modelled as an iterator over `Piece` structures to form a stream of tokens
/// being output.
///
/// This is a recursive-descent parser for the sake of simplicity, and if
/// necessary there's probably lots of room for improvement performance-wise.
pub struct Parser<'a> {
    input: &'a str,
    cur: str::CharOffsets<'a>,
    /// Error messages accumulated during parsing
    pub errors: Vec<string::String>,
}

impl<'a> Iterator<Piece<'a>> for Parser<'a> {
    fn next(&mut self) -> Option<Piece<'a>> {
        match self.cur.clone().next() {
            Some((pos, '{')) => {
                self.cur.next();
                if self.consume('{') {
                    Some(String(self.string(pos + 1)))
                } else {
                    let ret = Some(NextArgument(self.argument()));
                    self.must_consume('}');
                    ret
                }
            }
            Some((pos, '}')) => {
                self.cur.next();
                if self.consume('}') {
                    Some(String(self.string(pos + 1)))
                } else {
                    self.err("unmatched `}` found");
                    None
                }
            }
            Some((pos, _)) => { Some(String(self.string(pos))) }
            None => None
        }
    }
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given format string
    pub fn new<'a>(s: &'a str) -> Parser<'a> {
        Parser {
            input: s,
            cur: s.char_indices(),
            errors: vec!(),
        }
    }

    /// Notifies of an error. The message doesn't actually need to be of type
    /// String, but I think it does when this eventually uses conditions so it
    /// might as well start using it now.
    fn err(&mut self, msg: &str) {
        self.errors.push(msg.to_string());
    }

    /// Optionally consumes the specified character. If the character is not at
    /// the current position, then the current iterator isn't moved and false is
    /// returned, otherwise the character is consumed and true is returned.
    fn consume(&mut self, c: char) -> bool {
        match self.cur.clone().next() {
            Some((_, maybe)) if c == maybe => {
                self.cur.next();
                true
            }
            Some(..) | None => false,
        }
    }

    /// Forces consumption of the specified character. If the character is not
    /// found, an error is emitted.
    fn must_consume(&mut self, c: char) {
        self.ws();
        match self.cur.clone().next() {
            Some((_, maybe)) if c == maybe => {
                self.cur.next();
            }
            Some((_, other)) => {
                self.err(format!("expected `{}`, found `{}`",
                                 c,
                                 other).as_slice());
            }
            None => {
                self.err(format!("expected `{}` but string was terminated",
                                 c).as_slice());
            }
        }
    }

    /// Consumes all whitespace characters until the first non-whitespace
    /// character
    fn ws(&mut self) {
        loop {
            match self.cur.clone().next() {
                Some((_, c)) if char::is_whitespace(c) => { self.cur.next(); }
                Some(..) | None => { return }
            }
        }
    }

    /// Parses all of a string which is to be considered a "raw literal" in a
    /// format string. This is everything outside of the braces.
    fn string(&mut self, start: uint) -> &'a str {
        loop {
            // we may not consume the character, so clone the iterator
            match self.cur.clone().next() {
                Some((pos, '}')) | Some((pos, '{')) => {
                    return self.input.slice(start, pos);
                }
                Some(..) => { self.cur.next(); }
                None => {
                    self.cur.next();
                    return self.input.slice(start, self.input.len());
                }
            }
        }
    }

    /// Parses an Argument structure, or what's contained within braces inside
    /// the format string
    fn argument(&mut self) -> Argument<'a> {
        Argument {
            position: self.position(),
            format: self.format(),
        }
    }

    /// Parses a positional argument for a format. This could either be an
    /// integer index of an argument, a named argument, or a blank string.
    fn position(&mut self) -> Position<'a> {
        match self.integer() {
            Some(i) => { ArgumentIs(i) }
            None => {
                match self.cur.clone().next() {
                    Some((_, c)) if char::is_alphabetic(c) => {
                        ArgumentNamed(self.word())
                    }
                    _ => ArgumentNext
                }
            }
        }
    }

    /// Parses a format specifier at the current position, returning all of the
    /// relevant information in the FormatSpec struct.
    fn format(&mut self) -> FormatSpec<'a> {
        let mut spec = FormatSpec {
            fill: None,
            align: AlignUnknown,
            flags: 0,
            precision: CountImplied,
            width: CountImplied,
            ty: self.input.slice(0, 0),
        };
        if !self.consume(':') { return spec }

        // fill character
        match self.cur.clone().next() {
            Some((_, c)) => {
                match self.cur.clone().skip(1).next() {
                    Some((_, '>')) | Some((_, '<')) | Some((_, '^')) => {
                        spec.fill = Some(c);
                        self.cur.next();
                    }
                    Some(..) | None => {}
                }
            }
            None => {}
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
            spec.flags |= 1 << (FlagSignPlus as uint);
        } else if self.consume('-') {
            spec.flags |= 1 << (FlagSignMinus as uint);
        }
        // Alternate marker
        if self.consume('#') {
            spec.flags |= 1 << (FlagAlternate as uint);
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
                spec.flags |= 1 << (FlagSignAwareZeroPad as uint);
            }
        }
        if !havewidth {
            spec.width = self.count();
        }
        if self.consume('.') {
            if self.consume('*') {
                spec.precision = CountIsNextParam;
            } else {
                spec.precision = self.count();
            }
        }
        // Finally the actual format specifier
        if self.consume('?') {
            spec.ty = "?";
        } else {
            spec.ty = self.word();
        }
        return spec;
    }

    /// Parses a Count parameter at the current position. This does not check
    /// for 'CountIsNextParam' because that is only used in precision, not
    /// width.
    fn count(&mut self) -> Count<'a> {
        match self.integer() {
            Some(i) => {
                if self.consume('$') {
                    CountIsParam(i)
                } else {
                    CountIs(i)
                }
            }
            None => {
                let tmp = self.cur.clone();
                match self.word() {
                    word if word.len() > 0 => {
                        if self.consume('$') {
                            CountIsName(word)
                        } else {
                            self.cur = tmp;
                            CountImplied
                        }
                    }
                    _ => {
                        self.cur = tmp;
                        CountImplied
                    }
                }
            }
        }
    }

    /// Parses a word starting at the current position. A word is considered to
    /// be an alphabetic character followed by any number of alphanumeric
    /// characters.
    fn word(&mut self) -> &'a str {
        let start = match self.cur.clone().next() {
            Some((pos, c)) if char::is_XID_start(c) => {
                self.cur.next();
                pos
            }
            Some(..) | None => { return self.input.slice(0, 0); }
        };
        let mut end;
        loop {
            match self.cur.clone().next() {
                Some((_, c)) if char::is_XID_continue(c) => {
                    self.cur.next();
                }
                Some((pos, _)) => { end = pos; break }
                None => { end = self.input.len(); break }
            }
        }
        self.input.slice(start, end)
    }

    /// Optionally parses an integer at the current position. This doesn't deal
    /// with overflow at all, it's just accumulating digits.
    fn integer(&mut self) -> Option<uint> {
        let mut cur = 0;
        let mut found = false;
        loop {
            match self.cur.clone().next() {
                Some((_, c)) => {
                    match char::to_digit(c, 10) {
                        Some(i) => {
                            cur = cur * 10 + i;
                            found = true;
                            self.cur.next();
                        }
                        None => { break }
                    }
                }
                None => { break }
            }
        }
        if found {
            return Some(cur);
        } else {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(fmt: &'static str, p: &[Piece<'static>]) {
        let mut parser = Parser::new(fmt);
        assert!(p == parser.collect::<Vec<Piece<'static>>>().as_slice());
    }

    fn fmtdflt() -> FormatSpec<'static> {
        return FormatSpec {
            fill: None,
            align: AlignUnknown,
            flags: 0,
            precision: CountImplied,
            width: CountImplied,
            ty: "",
        }
    }

    fn musterr(s: &str) {
        let mut p = Parser::new(s);
        p.next();
        assert!(p.errors.len() != 0);
    }

    #[test]
    fn simple() {
        same("asdf", [String("asdf")]);
        same("a{{b", [String("a"), String("{b")]);
        same("a}}b", [String("a"), String("}b")]);
        same("a}}", [String("a"), String("}")]);
        same("}}", [String("}")]);
        same("\\}}", [String("\\"), String("}")]);
    }

    #[test] fn invalid01() { musterr("{") }
    #[test] fn invalid02() { musterr("}") }
    #[test] fn invalid04() { musterr("{3a}") }
    #[test] fn invalid05() { musterr("{:|}") }
    #[test] fn invalid06() { musterr("{:>>>}") }

    #[test]
    fn format_nothing() {
        same("{}", [NextArgument(Argument {
            position: ArgumentNext,
            format: fmtdflt(),
        })]);
    }
    #[test]
    fn format_position() {
        same("{3}", [NextArgument(Argument {
            position: ArgumentIs(3),
            format: fmtdflt(),
        })]);
    }
    #[test]
    fn format_position_nothing_else() {
        same("{3:}", [NextArgument(Argument {
            position: ArgumentIs(3),
            format: fmtdflt(),
        })]);
    }
    #[test]
    fn format_type() {
        same("{3:a}", [NextArgument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "a",
            },
        })]);
    }
    #[test]
    fn format_align_fill() {
        same("{3:>}", [NextArgument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: None,
                align: AlignRight,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "",
            },
        })]);
        same("{3:0<}", [NextArgument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: Some('0'),
                align: AlignLeft,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "",
            },
        })]);
        same("{3:*<abcd}", [NextArgument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: Some('*'),
                align: AlignLeft,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "abcd",
            },
        })]);
    }
    #[test]
    fn format_counts() {
        same("{:10s}", [NextArgument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountImplied,
                width: CountIs(10),
                ty: "s",
            },
        })]);
        same("{:10$.10s}", [NextArgument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIs(10),
                width: CountIsParam(10),
                ty: "s",
            },
        })]);
        same("{:.*s}", [NextArgument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIsNextParam,
                width: CountImplied,
                ty: "s",
            },
        })]);
        same("{:.10$s}", [NextArgument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIsParam(10),
                width: CountImplied,
                ty: "s",
            },
        })]);
        same("{:a$.b$s}", [NextArgument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIsName("b"),
                width: CountIsName("a"),
                ty: "s",
            },
        })]);
    }
    #[test]
    fn format_flags() {
        same("{:-}", [NextArgument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: (1 << FlagSignMinus as uint),
                precision: CountImplied,
                width: CountImplied,
                ty: "",
            },
        })]);
        same("{:+#}", [NextArgument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: (1 << FlagSignPlus as uint) | (1 << FlagAlternate as uint),
                precision: CountImplied,
                width: CountImplied,
                ty: "",
            },
        })]);
    }
    #[test]
    fn format_mixture() {
        same("abcd {3:a} efg", [String("abcd "), NextArgument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "a",
            },
        }), String(" efg")]);
    }
}
