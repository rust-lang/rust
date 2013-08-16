// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;

use char;
use str;
use iterator;

condition! { pub parse_error: ~str -> (); }

/// A piece is a portion of the format string which represents the next part to
/// emit. These are emitted as a stream by the `Parser` class.
#[deriving(Eq)]
pub enum Piece<'self> {
    /// A literal string which should directly be emitted
    String(&'self str),
    /// A back-reference to whatever the current argument is. This is used
    /// inside of a method call to refer back to the original argument.
    CurrentArgument,
    /// This describes that formatting should process the next argument (as
    /// specified inside) for emission.
    Argument(Argument<'self>),
}

/// Representation of an argument specification.
#[deriving(Eq)]
pub struct Argument<'self> {
    /// Where to find this argument
    position: Position<'self>,
    /// How to format the argument
    format: FormatSpec<'self>,
    /// If not `None`, what method to invoke on the argument
    method: Option<~Method<'self>>
}

/// Specification for the formatting of an argument in the format string.
#[deriving(Eq)]
pub struct FormatSpec<'self> {
    /// Optionally specified character to fill alignment with
    fill: Option<char>,
    /// Optionally specified alignment
    align: Alignment,
    /// Packed version of various flags provided
    flags: uint,
    /// The integer precision to use
    precision: Count,
    /// The string width requested for the resulting format
    width: Count,
    /// The descriptor string representing the name of the format desired for
    /// this argument, this can be empty or any number of characters, although
    /// it is required to be one word.
    ty: &'self str
}

/// Enum describing where an argument for a format can be located.
#[deriving(Eq)]
pub enum Position<'self> {
    ArgumentNext, ArgumentIs(uint), ArgumentNamed(&'self str)
}

/// Enum of alignments which are supported.
#[deriving(Eq)]
pub enum Alignment { AlignLeft, AlignRight, AlignUnknown }

/// Various flags which can be applied to format strings, the meaning of these
/// flags is defined by the formatters themselves.
#[deriving(Eq)]
pub enum Flag {
    FlagSignPlus,
    FlagSignMinus,
    FlagAlternate,
    FlagSignAwareZeroPad,
}

/// A count is used for the precision and width parameters of an integer, and
/// can reference either an argument or a literal integer.
#[deriving(Eq)]
pub enum Count {
    CountIs(uint),
    CountIsParam(uint),
    CountIsNextParam,
    CountImplied,
}

/// Enum describing all of the possible methods which the formatting language
/// currently supports.
#[deriving(Eq)]
pub enum Method<'self> {
    /// A plural method selects on an integer over a list of either integer or
    /// keyword-defined clauses. The meaning of the keywords is defined by the
    /// current locale.
    ///
    /// An offset is optionally present at the beginning which is used to match
    /// against keywords, but it is not matched against the literal integers.
    ///
    /// The final element of this enum is the default "other" case which is
    /// always required to be specified.
    Plural(Option<uint>, ~[PluralArm<'self>], ~[Piece<'self>]),

    /// A select method selects over a string. Each arm is a different string
    /// which can be selected for.
    ///
    /// As with `Plural`, a default "other" case is required as well.
    Select(~[SelectArm<'self>], ~[Piece<'self>]),
}

/// Structure representing one "arm" of the `plural` function.
#[deriving(Eq)]
pub struct PluralArm<'self> {
    /// A selector can either be specified by a keyword or with an integer
    /// literal.
    selector: Either<PluralKeyword, uint>,
    /// Array of pieces which are the format of this arm
    result: ~[Piece<'self>],
}

/// Enum of the 5 CLDR plural keywords. There is one more, "other", but that is
/// specially placed in the `Plural` variant of `Method`
///
/// http://www.icu-project.org/apiref/icu4c/classicu_1_1PluralRules.html
#[deriving(Eq, IterBytes)]
pub enum PluralKeyword {
    Zero, One, Two, Few, Many
}

/// Structure representing one "arm" of the `select` function.
#[deriving(Eq)]
pub struct SelectArm<'self> {
    /// String selector which guards this arm
    selector: &'self str,
    /// Array of pieces which are the format of this arm
    result: ~[Piece<'self>],
}

/// The parser structure for interpreting the input format string. This is
/// modelled as an iterator over `Piece` structures to form a stream of tokens
/// being output.
///
/// This is a recursive-descent parser for the sake of simplicity, and if
/// necessary there's probably lots of room for improvement performance-wise.
pub struct Parser<'self> {
    priv input: &'self str,
    priv cur: str::CharOffsetIterator<'self>,
}

impl<'self> iterator::Iterator<Piece<'self>> for Parser<'self> {
    fn next(&mut self) -> Option<Piece<'self>> {
        match self.cur.clone().next() {
            Some((_, '#')) => { self.cur.next(); Some(CurrentArgument) }
            Some((_, '{')) => {
                self.cur.next();
                let ret = Some(Argument(self.argument()));
                if !self.consume('}') {
                    self.err(~"unterminated format string");
                }
                ret
            }
            Some((pos, '\\')) => {
                self.cur.next();
                self.escape(); // ensure it's a valid escape sequence
                Some(String(self.string(pos + 1))) // skip the '\' character
            }
            Some((_, '}')) | None => { None }
            Some((pos, _)) => {
                Some(String(self.string(pos)))
            }
        }
    }
}

impl<'self> Parser<'self> {
    /// Creates a new parser for the given format string
    pub fn new<'a>(s: &'a str) -> Parser<'a> {
        Parser {
            input: s,
            cur: s.char_offset_iter(),
        }
    }

    /// Notifies of an error. The message doesn't actually need to be of type
    /// ~str, but I think it does when this eventually uses conditions so it
    /// might as well start using it now.
    fn err(&self, msg: ~str) {
        parse_error::cond.raise(msg);
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
            Some(*) | None => false,
        }
    }

    /// Attempts to consume any amount of whitespace followed by a character
    fn wsconsume(&mut self, c: char) -> bool {
        self.ws(); self.consume(c)
    }

    /// Consumes all whitespace characters until the first non-whitespace
    /// character
    fn ws(&mut self) {
        loop {
            match self.cur.clone().next() {
                Some((_, c)) if char::is_whitespace(c) => { self.cur.next(); }
                Some(*) | None => { return }
            }
        }
    }

    /// Consumes an escape sequence, failing if there is not a valid character
    /// to be escaped.
    fn escape(&mut self) -> char {
        match self.cur.next() {
            Some((_, c @ '#')) | Some((_, c @ '{')) |
            Some((_, c @ '\\')) | Some((_, c @ '}')) => { c }
            Some((_, c)) => {
                self.err(fmt!("invalid escape character `%c`", c));
                c
            }
            None => {
                self.err(~"expected an escape sequence, but format string was \
                           terminated");
                ' '
            }
        }
    }

    /// Parses all of a string which is to be considered a "raw literal" in a
    /// format string. This is everything outside of the braces.
    fn string(&mut self, start: uint) -> &'self str {
        loop {
            // we may not consume the character, so clone the iterator
            match self.cur.clone().next() {
                Some((pos, '\\')) | Some((pos, '#')) |
                Some((pos, '}')) | Some((pos, '{')) => {
                    return self.input.slice(start, pos);
                }
                Some(*) => { self.cur.next(); }
                None => {
                    self.cur.next();
                    return self.input.slice(start, self.input.len());
                }
            }
        }
    }

    /// Parses an Argument structure, or what's contained within braces inside
    /// the format string
    fn argument(&mut self) -> Argument<'self> {
        Argument {
            position: self.position(),
            format: self.format(),
            method: self.method(),
        }
    }

    /// Parses a positional argument for a format. This could either be an
    /// integer index of an argument, a named argument, or a blank string.
    fn position(&mut self) -> Position<'self> {
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
    fn format(&mut self) -> FormatSpec<'self> {
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
                    Some((_, '>')) | Some((_, '<')) => {
                        spec.fill = Some(c);
                        self.cur.next();
                    }
                    Some(*) | None => {}
                }
            }
            None => {}
        }
        // Alignment
        if self.consume('<') {
            spec.align = AlignLeft;
        } else if self.consume('>') {
            spec.align = AlignRight;
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
        if self.consume('0') {
            spec.flags |= 1 << (FlagSignAwareZeroPad as uint);
        }
        spec.width = self.count();
        if self.consume('.') {
            if self.consume('*') {
                spec.precision = CountIsNextParam;
            } else {
                spec.precision = self.count();
            }
        }
        // Finally the actual format specifier
        spec.ty = self.word();
        return spec;
    }

    /// Parses a method to be applied to the previously specified argument and
    /// its format. The two current supported methods are 'plural' and 'select'
    fn method(&mut self) -> Option<~Method<'self>> {
        if !self.wsconsume(',') {
            return None;
        }
        self.ws();
        match self.word() {
            "select" => {
                if !self.wsconsume(',') {
                    self.err(~"`select` must be followed by `,`");
                }
                Some(self.select())
            }
            "plural" => {
                if !self.wsconsume(',') {
                    self.err(~"`plural` must be followed by `,`");
                }
                Some(self.plural())
            }
            "" => {
                self.err(~"expected method after comma");
                return None;
            }
            method => {
                self.err(fmt!("unknown method: `%s`", method));
                return None;
            }
        }
    }

    /// Parses a 'select' statement (after the initial 'select' word)
    fn select(&mut self) -> ~Method<'self> {
        let mut other = None;
        let mut arms = ~[];
        // Consume arms one at a time
        loop {
            self.ws();
            let selector = self.word();
            if selector == "" {
                self.err(~"cannot have an empty selector");
                break
            }
            if !self.wsconsume('{') {
                self.err(~"selector must be followed by `{`");
            }
            let pieces = self.collect();
            if !self.wsconsume('}') {
                self.err(~"selector case must be terminated by `}`");
            }
            if selector == "other" {
                if !other.is_none() {
                    self.err(~"multiple `other` statements in `select");
                }
                other = Some(pieces);
            } else {
                arms.push(SelectArm { selector: selector, result: pieces });
            }
            self.ws();
            match self.cur.clone().next() {
                Some((_, '}')) => { break }
                Some(*) | None => {}
            }
        }
        // The "other" selector must be present
        let other = match other {
            Some(arm) => { arm }
            None => {
                self.err(~"`select` statement must provide an `other` case");
                ~[]
            }
        };
        ~Select(arms, other)
    }

    /// Parses a 'plural' statement (after the initial 'plural' word)
    fn plural(&mut self) -> ~Method<'self> {
        let mut offset = None;
        let mut other = None;
        let mut arms = ~[];

        // First, attempt to parse the 'offset:' field. We know the set of
        // selector words which can appear in plural arms, and the only ones
        // which start with 'o' are "other" and "offset", hence look two
        // characters deep to see if we can consume the word "offset"
        self.ws();
        let mut it = self.cur.clone();
        match it.next() {
            Some((_, 'o')) => {
                match it.next() {
                    Some((_, 'f')) => {
                        let word = self.word();
                        if word != "offset" {
                            self.err(fmt!("expected `offset`, found `%s`",
                                          word));
                        } else {
                            if !self.consume(':') {
                                self.err(~"`offset` must be followed by `:`");
                            }
                            match self.integer() {
                                Some(i) => { offset = Some(i); }
                                None => {
                                    self.err(~"offset must be an integer");
                                }
                            }
                        }
                    }
                    Some(*) | None => {}
                }
            }
            Some(*) | None => {}
        }

        // Next, generate all the arms
        loop {
            let mut isother = false;
            let selector = if self.wsconsume('=') {
                match self.integer() {
                    Some(i) => Right(i),
                    None => {
                        self.err(~"plural `=` selectors must be followed by an \
                                   integer");
                        Right(0)
                    }
                }
            } else {
                let word = self.word();
                match word {
                    "other" => { isother = true; Left(Zero) }
                    "zero"  => Left(Zero),
                    "one"   => Left(One),
                    "two"   => Left(Two),
                    "few"   => Left(Few),
                    "many"  => Left(Many),
                    word    => {
                        self.err(fmt!("unexpected plural selector `%s`", word));
                        if word == "" {
                            break
                        } else {
                            Left(Zero)
                        }
                    }
                }
            };
            if !self.wsconsume('{') {
                self.err(~"selector must be followed by `{`");
            }
            let pieces = self.collect();
            if !self.wsconsume('}') {
                self.err(~"selector case must be terminated by `}`");
            }
            if isother {
                if !other.is_none() {
                    self.err(~"multiple `other` statements in `select");
                }
                other = Some(pieces);
            } else {
                arms.push(PluralArm { selector: selector, result: pieces });
            }
            self.ws();
            match self.cur.clone().next() {
                Some((_, '}')) => { break }
                Some(*) | None => {}
            }
        }

        let other = match other {
            Some(arm) => { arm }
            None => {
                self.err(~"`plural` statement must provide an `other` case");
                ~[]
            }
        };
        ~Plural(offset, arms, other)
    }

    /// Parses a Count parameter at the current position. This does not check
    /// for 'CountIsNextParam' because that is only used in precision, not
    /// width.
    fn count(&mut self) -> Count {
        match self.integer() {
            Some(i) => {
                if self.consume('$') {
                    CountIsParam(i)
                } else {
                    CountIs(i)
                }
            }
            None => { CountImplied }
        }
    }

    /// Parses a word starting at the current position. A word is considered to
    /// be an alphabetic character followed by any number of alphanumeric
    /// characters.
    fn word(&mut self) -> &'self str {
        let start = match self.cur.clone().next() {
            Some((pos, c)) if char::is_alphabetic(c) => {
                self.cur.next();
                pos
            }
            Some(*) | None => { return self.input.slice(0, 0); }
        };
        let mut end;
        loop {
            match self.cur.clone().next() {
                Some((_, c)) if char::is_alphanumeric(c) => {
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
    use prelude::*;
    use realstd::fmt::{String};

    fn same(fmt: &'static str, p: ~[Piece<'static>]) {
        let mut parser = Parser::new(fmt);
        assert_eq!(p, parser.collect());
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
        Parser::new(s).next();
    }

    #[test]
    fn simple() {
        same("asdf", ~[String("asdf")]);
        same("a\\{b", ~[String("a"), String("{b")]);
        same("a\\#b", ~[String("a"), String("#b")]);
        same("a\\}b", ~[String("a"), String("}b")]);
        same("a\\}", ~[String("a"), String("}")]);
        same("\\}", ~[String("}")]);
    }

    #[test] #[should_fail] fn invalid01() { musterr("{") }
    #[test] #[should_fail] fn invalid02() { musterr("\\") }
    #[test] #[should_fail] fn invalid03() { musterr("\\a") }
    #[test] #[should_fail] fn invalid04() { musterr("{3a}") }
    #[test] #[should_fail] fn invalid05() { musterr("{:|}") }
    #[test] #[should_fail] fn invalid06() { musterr("{:>>>}") }

    #[test]
    fn format_nothing() {
        same("{}", ~[Argument(Argument {
            position: ArgumentNext,
            format: fmtdflt(),
            method: None,
        })]);
    }
    #[test]
    fn format_position() {
        same("{3}", ~[Argument(Argument {
            position: ArgumentIs(3),
            format: fmtdflt(),
            method: None,
        })]);
    }
    #[test]
    fn format_position_nothing_else() {
        same("{3:}", ~[Argument(Argument {
            position: ArgumentIs(3),
            format: fmtdflt(),
            method: None,
        })]);
    }
    #[test]
    fn format_type() {
        same("{3:a}", ~[Argument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "a",
            },
            method: None,
        })]);
    }
    #[test]
    fn format_align_fill() {
        same("{3:>}", ~[Argument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: None,
                align: AlignRight,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "",
            },
            method: None,
        })]);
        same("{3:0<}", ~[Argument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: Some('0'),
                align: AlignLeft,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "",
            },
            method: None,
        })]);
        same("{3:*<abcd}", ~[Argument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: Some('*'),
                align: AlignLeft,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "abcd",
            },
            method: None,
        })]);
    }
    #[test]
    fn format_counts() {
        same("{:10s}", ~[Argument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountImplied,
                width: CountIs(10),
                ty: "s",
            },
            method: None,
        })]);
        same("{:10$.10s}", ~[Argument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIs(10),
                width: CountIsParam(10),
                ty: "s",
            },
            method: None,
        })]);
        same("{:.*s}", ~[Argument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIsNextParam,
                width: CountImplied,
                ty: "s",
            },
            method: None,
        })]);
        same("{:.10$s}", ~[Argument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIsParam(10),
                width: CountImplied,
                ty: "s",
            },
            method: None,
        })]);
    }
    #[test]
    fn format_flags() {
        same("{:-}", ~[Argument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: (1 << FlagSignMinus as uint),
                precision: CountImplied,
                width: CountImplied,
                ty: "",
            },
            method: None,
        })]);
        same("{:+#}", ~[Argument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: (1 << FlagSignPlus as uint) | (1 << FlagAlternate as uint),
                precision: CountImplied,
                width: CountImplied,
                ty: "",
            },
            method: None,
        })]);
    }
    #[test]
    fn format_mixture() {
        same("abcd {3:a} efg", ~[String("abcd "), Argument(Argument {
            position: ArgumentIs(3),
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                ty: "a",
            },
            method: None,
        }), String(" efg")]);
    }

    #[test]
    fn select_simple() {
        same("{, select, other { haha } }", ~[Argument(Argument{
            position: ArgumentNext,
            format: fmtdflt(),
            method: Some(~Select(~[], ~[String(" haha ")]))
        })]);
        same("{1, select, other { haha } }", ~[Argument(Argument{
            position: ArgumentIs(1),
            format: fmtdflt(),
            method: Some(~Select(~[], ~[String(" haha ")]))
        })]);
        same("{1, select, other {#} }", ~[Argument(Argument{
            position: ArgumentIs(1),
            format: fmtdflt(),
            method: Some(~Select(~[], ~[CurrentArgument]))
        })]);
        same("{1, select, other {{2, select, other {lol}}} }", ~[Argument(Argument{
            position: ArgumentIs(1),
            format: fmtdflt(),
            method: Some(~Select(~[], ~[Argument(Argument{
                position: ArgumentIs(2),
                format: fmtdflt(),
                method: Some(~Select(~[], ~[String("lol")]))
            })])) // wat
        })]);
    }

    #[test]
    fn select_cases() {
        same("{1, select, a{1} b{2} c{3} other{4} }", ~[Argument(Argument{
            position: ArgumentIs(1),
            format: fmtdflt(),
            method: Some(~Select(~[
                SelectArm{ selector: "a", result: ~[String("1")] },
                SelectArm{ selector: "b", result: ~[String("2")] },
                SelectArm{ selector: "c", result: ~[String("3")] },
            ], ~[String("4")]))
        })]);
    }

    #[test] #[should_fail] fn badselect01() {
        musterr("{select, }")
    }
    #[test] #[should_fail] fn badselect02() {
        musterr("{1, select}")
    }
    #[test] #[should_fail] fn badselect03() {
        musterr("{1, select, }")
    }
    #[test] #[should_fail] fn badselect04() {
        musterr("{1, select, a {}}")
    }
    #[test] #[should_fail] fn badselect05() {
        musterr("{1, select, other }}")
    }
    #[test] #[should_fail] fn badselect06() {
        musterr("{1, select, other {}")
    }
    #[test] #[should_fail] fn badselect07() {
        musterr("{select, other {}")
    }
    #[test] #[should_fail] fn badselect08() {
        musterr("{1 select, other {}")
    }
    #[test] #[should_fail] fn badselect09() {
        musterr("{:d select, other {}")
    }
    #[test] #[should_fail] fn badselect10() {
        musterr("{1:d select, other {}")
    }

    #[test]
    fn plural_simple() {
        same("{, plural, other { haha } }", ~[Argument(Argument{
            position: ArgumentNext,
            format: fmtdflt(),
            method: Some(~Plural(None, ~[], ~[String(" haha ")]))
        })]);
        same("{:, plural, other { haha } }", ~[Argument(Argument{
            position: ArgumentNext,
            format: fmtdflt(),
            method: Some(~Plural(None, ~[], ~[String(" haha ")]))
        })]);
        same("{, plural, offset:1 =2{2} =3{3} many{yes} other{haha} }",
        ~[Argument(Argument{
            position: ArgumentNext,
            format: fmtdflt(),
            method: Some(~Plural(Some(1), ~[
                PluralArm{ selector: Right(2), result: ~[String("2")] },
                PluralArm{ selector: Right(3), result: ~[String("3")] },
                PluralArm{ selector: Left(Many), result: ~[String("yes")] }
            ], ~[String("haha")]))
        })]);
    }
}
