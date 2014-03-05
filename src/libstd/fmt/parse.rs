// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Parsing of format strings
//!
//! These structures are used when parsing format strings for the compiler.
//! Parsing does not currently happen at runtime (structures of std::fmt::rt are
//! generated instead).

use prelude::*;

use char;
use str;

/// A piece is a portion of the format string which represents the next part to
/// emit. These are emitted as a stream by the `Parser` class.
#[deriving(Eq)]
pub enum Piece<'a> {
    /// A literal string which should directly be emitted
    String(&'a str),
    /// A back-reference to whatever the current argument is. This is used
    /// inside of a method call to refer back to the original argument.
    CurrentArgument,
    /// This describes that formatting should process the next argument (as
    /// specified inside) for emission.
    Argument(Argument<'a>),
}

/// Representation of an argument specification.
#[deriving(Eq)]
pub struct Argument<'a> {
    /// Where to find this argument
    position: Position<'a>,
    /// How to format the argument
    format: FormatSpec<'a>,
    /// If not `None`, what method to invoke on the argument
    method: Option<~Method<'a>>
}

/// Specification for the formatting of an argument in the format string.
#[deriving(Eq)]
pub struct FormatSpec<'a> {
    /// Optionally specified character to fill alignment with
    fill: Option<char>,
    /// Optionally specified alignment
    align: Alignment,
    /// Packed version of various flags provided
    flags: uint,
    /// The integer precision to use
    precision: Count<'a>,
    /// The string width requested for the resulting format
    width: Count<'a>,
    /// The descriptor string representing the name of the format desired for
    /// this argument, this can be empty or any number of characters, although
    /// it is required to be one word.
    ty: &'a str
}

/// Enum describing where an argument for a format can be located.
#[deriving(Eq)]
#[allow(missing_doc)]
pub enum Position<'a> {
    ArgumentNext, ArgumentIs(uint), ArgumentNamed(&'a str)
}

/// Enum of alignments which are supported.
#[deriving(Eq)]
#[allow(missing_doc)]
pub enum Alignment { AlignLeft, AlignRight, AlignUnknown }

/// Various flags which can be applied to format strings, the meaning of these
/// flags is defined by the formatters themselves.
#[deriving(Eq)]
#[allow(missing_doc)]
pub enum Flag {
    FlagSignPlus,
    FlagSignMinus,
    FlagAlternate,
    FlagSignAwareZeroPad,
}

/// A count is used for the precision and width parameters of an integer, and
/// can reference either an argument or a literal integer.
#[deriving(Eq)]
#[allow(missing_doc)]
pub enum Count<'a> {
    CountIs(uint),
    CountIsName(&'a str),
    CountIsParam(uint),
    CountIsNextParam,
    CountImplied,
}

/// Enum describing all of the possible methods which the formatting language
/// currently supports.
#[deriving(Eq)]
pub enum Method<'a> {
    /// A plural method selects on an integer over a list of either integer or
    /// keyword-defined clauses. The meaning of the keywords is defined by the
    /// current locale.
    ///
    /// An offset is optionally present at the beginning which is used to match
    /// against keywords, but it is not matched against the literal integers.
    ///
    /// The final element of this enum is the default "other" case which is
    /// always required to be specified.
    Plural(Option<uint>, ~[PluralArm<'a>], ~[Piece<'a>]),

    /// A select method selects over a string. Each arm is a different string
    /// which can be selected for.
    ///
    /// As with `Plural`, a default "other" case is required as well.
    Select(~[SelectArm<'a>], ~[Piece<'a>]),
}

/// A selector for what pluralization a plural method should take
#[deriving(Eq, Hash)]
pub enum PluralSelector {
    /// One of the plural keywords should be used
    Keyword(PluralKeyword),
    /// A literal pluralization should be used
    Literal(uint),
}

/// Structure representing one "arm" of the `plural` function.
#[deriving(Eq)]
pub struct PluralArm<'a> {
    /// A selector can either be specified by a keyword or with an integer
    /// literal.
    selector: PluralSelector,
    /// Array of pieces which are the format of this arm
    result: ~[Piece<'a>],
}

/// Enum of the 5 CLDR plural keywords. There is one more, "other", but that is
/// specially placed in the `Plural` variant of `Method`
///
/// http://www.icu-project.org/apiref/icu4c/classicu_1_1PluralRules.html
#[deriving(Eq, Hash)]
#[allow(missing_doc)]
pub enum PluralKeyword {
    Zero, One, Two, Few, Many
}

/// Structure representing one "arm" of the `select` function.
#[deriving(Eq)]
pub struct SelectArm<'a> {
    /// String selector which guards this arm
    selector: &'a str,
    /// Array of pieces which are the format of this arm
    result: ~[Piece<'a>],
}

/// The parser structure for interpreting the input format string. This is
/// modelled as an iterator over `Piece` structures to form a stream of tokens
/// being output.
///
/// This is a recursive-descent parser for the sake of simplicity, and if
/// necessary there's probably lots of room for improvement performance-wise.
pub struct Parser<'a> {
    priv input: &'a str,
    priv cur: str::CharOffsets<'a>,
    priv depth: uint,
    /// Error messages accumulated during parsing
    errors: ~[~str],
}

impl<'a> Iterator<Piece<'a>> for Parser<'a> {
    fn next(&mut self) -> Option<Piece<'a>> {
        match self.cur.clone().next() {
            Some((_, '#')) => { self.cur.next(); Some(CurrentArgument) }
            Some((_, '{')) => {
                self.cur.next();
                let ret = Some(Argument(self.argument()));
                self.must_consume('}');
                ret
            }
            Some((pos, '\\')) => {
                self.cur.next();
                self.escape(); // ensure it's a valid escape sequence
                Some(String(self.string(pos + 1))) // skip the '\' character
            }
            Some((_, '}')) if self.depth == 0 => {
                self.cur.next();
                self.err("unmatched `}` found");
                None
            }
            Some((_, '}')) | None => { None }
            Some((pos, _)) => {
                Some(String(self.string(pos)))
            }
        }
    }
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given format string
    pub fn new<'a>(s: &'a str) -> Parser<'a> {
        Parser {
            input: s,
            cur: s.char_indices(),
            depth: 0,
            errors: ~[],
        }
    }

    /// Notifies of an error. The message doesn't actually need to be of type
    /// ~str, but I think it does when this eventually uses conditions so it
    /// might as well start using it now.
    fn err(&mut self, msg: &str) {
        self.errors.push(msg.to_owned());
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
                self.err(
                    format!("expected `{}` but found `{}`", c, other));
            }
            None => {
                self.err(
                    format!("expected `{}` but string was terminated", c));
            }
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
                Some(..) | None => { return }
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
                self.err(format!("invalid escape character `{}`", c));
                c
            }
            None => {
                self.err("expected an escape sequence, but format string was \
                           terminated");
                ' '
            }
        }
    }

    /// Parses all of a string which is to be considered a "raw literal" in a
    /// format string. This is everything outside of the braces.
    fn string(&mut self, start: uint) -> &'a str {
        loop {
            // we may not consume the character, so clone the iterator
            match self.cur.clone().next() {
                Some((pos, '\\')) | Some((pos, '#')) |
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
            method: self.method(),
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
                    Some((_, '>')) | Some((_, '<')) => {
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

    /// Parses a method to be applied to the previously specified argument and
    /// its format. The two current supported methods are 'plural' and 'select'
    fn method(&mut self) -> Option<~Method<'a>> {
        if !self.wsconsume(',') {
            return None;
        }
        self.ws();
        match self.word() {
            "select" => {
                self.must_consume(',');
                Some(self.select())
            }
            "plural" => {
                self.must_consume(',');
                Some(self.plural())
            }
            "" => {
                self.err("expected method after comma");
                return None;
            }
            method => {
                self.err(format!("unknown method: `{}`", method));
                return None;
            }
        }
    }

    /// Parses a 'select' statement (after the initial 'select' word)
    fn select(&mut self) -> ~Method<'a> {
        let mut other = None;
        let mut arms = ~[];
        // Consume arms one at a time
        loop {
            self.ws();
            let selector = self.word();
            if selector == "" {
                self.err("cannot have an empty selector");
                break
            }
            self.must_consume('{');
            self.depth += 1;
            let pieces = self.collect();
            self.depth -= 1;
            self.must_consume('}');
            if selector == "other" {
                if !other.is_none() {
                    self.err("multiple `other` statements in `select");
                }
                other = Some(pieces);
            } else {
                arms.push(SelectArm { selector: selector, result: pieces });
            }
            self.ws();
            match self.cur.clone().next() {
                Some((_, '}')) => { break }
                Some(..) | None => {}
            }
        }
        // The "other" selector must be present
        let other = match other {
            Some(arm) => { arm }
            None => {
                self.err("`select` statement must provide an `other` case");
                ~[]
            }
        };
        ~Select(arms, other)
    }

    /// Parses a 'plural' statement (after the initial 'plural' word)
    fn plural(&mut self) -> ~Method<'a> {
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
                            self.err(format!("expected `offset`, found `{}`",
                                             word));
                        } else {
                            self.must_consume(':');
                            match self.integer() {
                                Some(i) => { offset = Some(i); }
                                None => {
                                    self.err("offset must be an integer");
                                }
                            }
                        }
                    }
                    Some(..) | None => {}
                }
            }
            Some(..) | None => {}
        }

        // Next, generate all the arms
        loop {
            let mut isother = false;
            let selector = if self.wsconsume('=') {
                match self.integer() {
                    Some(i) => Literal(i),
                    None => {
                        self.err("plural `=` selectors must be followed by an \
                                  integer");
                        Literal(0)
                    }
                }
            } else {
                let word = self.word();
                match word {
                    "other" => { isother = true; Keyword(Zero) }
                    "zero"  => Keyword(Zero),
                    "one"   => Keyword(One),
                    "two"   => Keyword(Two),
                    "few"   => Keyword(Few),
                    "many"  => Keyword(Many),
                    word    => {
                        self.err(format!("unexpected plural selector `{}`",
                                         word));
                        if word == "" {
                            break
                        } else {
                            Keyword(Zero)
                        }
                    }
                }
            };
            self.must_consume('{');
            self.depth += 1;
            let pieces = self.collect();
            self.depth -= 1;
            self.must_consume('}');
            if isother {
                if !other.is_none() {
                    self.err("multiple `other` statements in `select");
                }
                other = Some(pieces);
            } else {
                arms.push(PluralArm { selector: selector, result: pieces });
            }
            self.ws();
            match self.cur.clone().next() {
                Some((_, '}')) => { break }
                Some(..) | None => {}
            }
        }

        let other = match other {
            Some(arm) => { arm }
            None => {
                self.err("`plural` statement must provide an `other` case");
                ~[]
            }
        };
        ~Plural(offset, arms, other)
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
                    word if word.len() > 0 && self.consume('$') => {
                        CountIsName(word)
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
    use prelude::*;

    fn same(fmt: &'static str, p: ~[Piece<'static>]) {
        let mut parser = Parser::new(fmt);
        assert!(p == parser.collect());
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
        same("asdf", ~[String("asdf")]);
        same("a\\{b", ~[String("a"), String("{b")]);
        same("a\\#b", ~[String("a"), String("#b")]);
        same("a\\}b", ~[String("a"), String("}b")]);
        same("a\\}", ~[String("a"), String("}")]);
        same("\\}", ~[String("}")]);
    }

    #[test] fn invalid01() { musterr("{") }
    #[test] fn invalid02() { musterr("\\") }
    #[test] fn invalid03() { musterr("\\a") }
    #[test] fn invalid04() { musterr("{3a}") }
    #[test] fn invalid05() { musterr("{:|}") }
    #[test] fn invalid06() { musterr("{:>>>}") }

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
        same("{:a$.b$s}", ~[Argument(Argument {
            position: ArgumentNext,
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIsName("b"),
                width: CountIsName("a"),
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

    #[test] fn badselect01() { musterr("{select, }") }
    #[test] fn badselect02() { musterr("{1, select}") }
    #[test] fn badselect03() { musterr("{1, select, }") }
    #[test] fn badselect04() { musterr("{1, select, a {}}") }
    #[test] fn badselect05() { musterr("{1, select, other }}") }
    #[test] fn badselect06() { musterr("{1, select, other {}") }
    #[test] fn badselect07() { musterr("{select, other {}") }
    #[test] fn badselect08() { musterr("{1 select, other {}") }
    #[test] fn badselect09() { musterr("{:d select, other {}") }
    #[test] fn badselect10() { musterr("{1:d select, other {}") }

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
                PluralArm{ selector: Literal(2), result: ~[String("2")] },
                PluralArm{ selector: Literal(3), result: ~[String("3")] },
                PluralArm{ selector: Keyword(Many), result: ~[String("yes")] }
            ], ~[String("haha")]))
        })]);
    }
}
