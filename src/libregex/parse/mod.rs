// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::char;
use std::cmp;
use std::fmt;
use std::iter;
use std::num;
use std::str;

/// Static data containing Unicode ranges for general categories and scripts.
use self::unicode::{UNICODE_CLASSES, PERLD, PERLS, PERLW};
#[allow(visible_private_types)]
pub mod unicode;

/// The maximum number of repetitions allowed with the `{n,m}` syntax.
static MAX_REPEAT: uint = 1000;

/// Error corresponds to something that can go wrong while parsing
/// a regular expression.
///
/// (Once an expression is compiled, it is not possible to produce an error
/// via searching, splitting or replacing.)
pub struct Error {
    /// The *approximate* character index of where the error occurred.
    pub pos: uint,
    /// A message describing the error.
    pub msg: String,
}

impl fmt::Show for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Regex syntax error near position {}: {}",
               self.pos, self.msg)
    }
}

/// Represents the abstract syntax of a regular expression.
/// It is showable so that error messages resulting from a bug can provide
/// useful information.
/// It is cloneable so that expressions can be repeated for the counted
/// repetition feature. (No other copying is done.)
///
/// Note that this representation prevents one from reproducing the regex as
/// it was typed. (But it could be used to reproduce an equivalent regex.)
#[deriving(Show, Clone)]
pub enum Ast {
    Nothing,
    Literal(char, Flags),
    Dot(Flags),
    Class(Vec<(char, char)>, Flags),
    Begin(Flags),
    End(Flags),
    WordBoundary(Flags),
    Capture(uint, Option<String>, Box<Ast>),
    // Represent concatenation as a flat vector to avoid blowing the
    // stack in the compiler.
    Cat(Vec<Ast>),
    Alt(Box<Ast>, Box<Ast>),
    Rep(Box<Ast>, Repeater, Greed),
}

#[deriving(Show, PartialEq, Clone)]
pub enum Repeater {
    ZeroOne,
    ZeroMore,
    OneMore,
}

#[deriving(Show, Clone)]
pub enum Greed {
    Greedy,
    Ungreedy,
}

impl Greed {
    pub fn is_greedy(&self) -> bool {
        match *self {
            Greedy => true,
            _ => false,
        }
    }

    fn swap(self, swapped: bool) -> Greed {
        if !swapped { return self }
        match self {
            Greedy => Ungreedy,
            Ungreedy => Greedy,
        }
    }
}

/// BuildAst is a regrettable type that represents intermediate state for
/// constructing an abstract syntax tree. Its central purpose is to facilitate
/// parsing groups and alternations while also maintaining a stack of flag
/// state.
#[deriving(Show)]
enum BuildAst {
    Ast(Ast),
    Paren(Flags, uint, String), // '('
    Bar, // '|'
}

impl BuildAst {
    fn paren(&self) -> bool {
        match *self {
            Paren(_, _, _) => true,
            _ => false,
        }
    }

    fn flags(&self) -> Flags {
        match *self {
            Paren(flags, _, _) => flags,
            _ => fail!("Cannot get flags from {}", self),
        }
    }

    fn capture(&self) -> Option<uint> {
        match *self {
            Paren(_, 0, _) => None,
            Paren(_, c, _) => Some(c),
            _ => fail!("Cannot get capture group from {}", self),
        }
    }

    fn capture_name(&self) -> Option<String> {
        match *self {
            Paren(_, 0, _) => None,
            Paren(_, _, ref name) => {
                if name.len() == 0 {
                    None
                } else {
                    Some(name.clone())
                }
            }
            _ => fail!("Cannot get capture name from {}", self),
        }
    }

    fn bar(&self) -> bool {
        match *self {
            Bar => true,
            _ => false,
        }
    }

    fn unwrap(self) -> Result<Ast, Error> {
        match self {
            Ast(x) => Ok(x),
            _ => fail!("Tried to unwrap non-AST item: {}", self),
        }
    }
}

/// Flags represents all options that can be twiddled by a user in an
/// expression.
pub type Flags = u8;

pub static FLAG_EMPTY:      u8 = 0;
pub static FLAG_NOCASE:     u8 = 1 << 0; // i
pub static FLAG_MULTI:      u8 = 1 << 1; // m
pub static FLAG_DOTNL:      u8 = 1 << 2; // s
pub static FLAG_SWAP_GREED: u8 = 1 << 3; // U
pub static FLAG_NEGATED:    u8 = 1 << 4; // char class or not word boundary

struct Parser<'a> {
    // The input, parsed only as a sequence of UTF8 code points.
    chars: Vec<char>,
    // The index of the current character in the input.
    chari: uint,
    // The intermediate state representing the AST.
    stack: Vec<BuildAst>,
    // The current set of flags.
    flags: Flags,
    // The total number of capture groups.
    // Incremented each time an opening left paren is seen (assuming it is
    // opening a capture group).
    caps: uint,
    // A set of all capture group names used only to detect duplicates.
    names: Vec<String>,
}

pub fn parse(s: &str) -> Result<Ast, Error> {
    Parser {
        chars: s.chars().collect(),
        chari: 0,
        stack: vec!(),
        flags: FLAG_EMPTY,
        caps: 0,
        names: vec!(),
    }.parse()
}

impl<'a> Parser<'a> {
    fn parse(&mut self) -> Result<Ast, Error> {
        if self.chars.len() == 0 {
            return Ok(Nothing);
        }
        loop {
            let c = self.cur();
            match c {
                '?' | '*' | '+' => try!(self.push_repeater(c)),
                '\\' => {
                    let ast = try!(self.parse_escape());
                    self.push(ast)
                }
                '{' => try!(self.parse_counted()),
                '[' => match self.try_parse_ascii() {
                    None => try!(self.parse_class()),
                    Some(class) => self.push(class),
                },
                '(' => {
                    if self.peek_is(1, '?') {
                        try!(self.expect('?'))
                        try!(self.parse_group_opts())
                    } else {
                        self.caps += 1;
                        self.stack.push(Paren(self.flags,
                                              self.caps,
                                              "".to_string()))
                    }
                }
                ')' => {
                    let catfrom = try!(
                        self.pos_last(false, |x| x.paren() || x.bar()));
                    try!(self.concat(catfrom));

                    let altfrom = try!(self.pos_last(false, |x| x.paren()));
                    // Before we smush the alternates together and pop off the
                    // left paren, let's grab the old flags and see if we
                    // need a capture.
                    let (cap, cap_name, oldflags) = {
                        let paren = self.stack.get(altfrom-1);
                        (paren.capture(), paren.capture_name(), paren.flags())
                    };
                    try!(self.alternate(altfrom));
                    self.flags = oldflags;

                    // If this was a capture, pop what we just pushed in
                    // alternate and make it a capture.
                    if cap.is_some() {
                        let ast = try!(self.pop_ast());
                        self.push(Capture(cap.unwrap(), cap_name, box ast));
                    }
                }
                '|' => {
                    let catfrom = try!(
                        self.pos_last(true, |x| x.paren() || x.bar()));
                    try!(self.concat(catfrom));

                    self.stack.push(Bar);
                }
                _ => try!(self.push_literal(c)),
            }
            if !self.next_char() {
                break
            }
        }

        // Try to improve error handling. At this point, there should be
        // no remaining open parens.
        if self.stack.iter().any(|x| x.paren()) {
            return self.err("Unclosed parenthesis.")
        }
        let catfrom = try!(self.pos_last(true, |x| x.bar()));
        try!(self.concat(catfrom));
        try!(self.alternate(0));

        assert!(self.stack.len() == 1);
        self.pop_ast()
    }

    fn noteof(&mut self, expected: &str) -> Result<(), Error> {
        match self.next_char() {
            true => Ok(()),
            false => {
                self.err(format!("Expected {} but got EOF.",
                                 expected).as_slice())
            }
        }
    }

    fn expect(&mut self, expected: char) -> Result<(), Error> {
        match self.next_char() {
            true if self.cur() == expected => Ok(()),
            true => self.err(format!("Expected '{}' but got '{}'.",
                                     expected, self.cur()).as_slice()),
            false => {
                self.err(format!("Expected '{}' but got EOF.",
                                 expected).as_slice())
            }
        }
    }

    fn next_char(&mut self) -> bool {
        self.chari += 1;
        self.chari < self.chars.len()
    }

    fn pop_ast(&mut self) -> Result<Ast, Error> {
        match self.stack.pop().unwrap().unwrap() {
            Err(e) => Err(e),
            Ok(ast) => Ok(ast),
        }
    }

    fn push(&mut self, ast: Ast) {
        self.stack.push(Ast(ast))
    }

    fn push_repeater(&mut self, c: char) -> Result<(), Error> {
        if self.stack.len() == 0 {
            return self.err(
                "A repeat operator must be preceded by a valid expression.")
        }
        let rep: Repeater = match c {
            '?' => ZeroOne, '*' => ZeroMore, '+' => OneMore,
            _ => fail!("Not a valid repeater operator."),
        };

        match self.peek(1) {
            Some('*') | Some('+') =>
                return self.err(
                    "Double repeat operators are not supported."),
            _ => {},
        }
        let ast = try!(self.pop_ast());
        match ast {
            Begin(_) | End(_) | WordBoundary(_) =>
                return self.err(
                    "Repeat arguments cannot be empty width assertions."),
            _ => {}
        }
        let greed = try!(self.get_next_greedy());
        self.push(Rep(box ast, rep, greed));
        Ok(())
    }

    fn push_literal(&mut self, c: char) -> Result<(), Error> {
        match c {
            '.' => {
                self.push(Dot(self.flags))
            }
            '^' => {
                self.push(Begin(self.flags))
            }
            '$' => {
                self.push(End(self.flags))
            }
            _ => {
                self.push(Literal(c, self.flags))
            }
        }
        Ok(())
    }

    // Parses all forms of character classes.
    // Assumes that '[' is the current character.
    fn parse_class(&mut self) -> Result<(), Error> {
        let negated =
            if self.peek_is(1, '^') {
                try!(self.expect('^'))
                FLAG_NEGATED
            } else {
                FLAG_EMPTY
            };
        let mut ranges: Vec<(char, char)> = vec!();
        let mut alts: Vec<Ast> = vec!();

        if self.peek_is(1, ']') {
            try!(self.expect(']'))
            ranges.push((']', ']'))
        }
        while self.peek_is(1, '-') {
            try!(self.expect('-'))
            ranges.push(('-', '-'))
        }
        loop {
            try!(self.noteof("a closing ']' or a non-empty character class)"))
            let mut c = self.cur();
            match c {
                '[' =>
                    match self.try_parse_ascii() {
                        Some(Class(asciis, flags)) => {
                            alts.push(Class(asciis, flags ^ negated));
                            continue
                        }
                        Some(ast) =>
                            fail!("Expected Class AST but got '{}'", ast),
                        // Just drop down and try to add as a regular character.
                        None => {},
                    },
                '\\' => {
                    match try!(self.parse_escape()) {
                        Class(asciis, flags) => {
                            alts.push(Class(asciis, flags ^ negated));
                            continue
                        }
                        Literal(c2, _) => c = c2, // process below
                        Begin(_) | End(_) | WordBoundary(_) =>
                            return self.err(
                                "\\A, \\z, \\b and \\B are not valid escape \
                                 sequences inside a character class."),
                        ast => fail!("Unexpected AST item '{}'", ast),
                    }
                }
                _ => {},
            }
            match c {
                ']' => {
                    if ranges.len() > 0 {
                        let flags = negated | (self.flags & FLAG_NOCASE);
                        let mut ast = Class(combine_ranges(ranges), flags);
                        for alt in alts.move_iter() {
                            ast = Alt(box alt, box ast)
                        }
                        self.push(ast);
                    } else if alts.len() > 0 {
                        let mut ast = alts.pop().unwrap();
                        for alt in alts.move_iter() {
                            ast = Alt(box alt, box ast)
                        }
                        self.push(ast);
                    }
                    return Ok(())
                }
                c => {
                    if self.peek_is(1, '-') && !self.peek_is(2, ']') {
                        try!(self.expect('-'))
                        try!(self.noteof("not a ']'"))
                        let c2 = self.cur();
                        if c2 < c {
                            return self.err(format!("Invalid character class \
                                                     range '{}-{}'",
                                                    c,
                                                    c2).as_slice())
                        }
                        ranges.push((c, self.cur()))
                    } else {
                        ranges.push((c, c))
                    }
                }
            }
        }
    }

    // Tries to parse an ASCII character class of the form [:name:].
    // If successful, returns an AST character class corresponding to name
    // and moves the parser to the final ']' character.
    // If unsuccessful, no state is changed and None is returned.
    // Assumes that '[' is the current character.
    fn try_parse_ascii(&mut self) -> Option<Ast> {
        if !self.peek_is(1, ':') {
            return None
        }
        let closer =
            match self.pos(']') {
                Some(i) => i,
                None => return None,
            };
        if *self.chars.get(closer-1) != ':' {
            return None
        }
        if closer - self.chari <= 3 {
            return None
        }
        let mut name_start = self.chari + 2;
        let negated =
            if self.peek_is(2, '^') {
                name_start += 1;
                FLAG_NEGATED
            } else {
                FLAG_EMPTY
            };
        let name = self.slice(name_start, closer - 1);
        match find_class(ASCII_CLASSES, name.as_slice()) {
            None => None,
            Some(ranges) => {
                self.chari = closer;
                let flags = negated | (self.flags & FLAG_NOCASE);
                Some(Class(combine_ranges(ranges), flags))
            }
        }
    }

    // Parses counted repetition. Supports:
    // {n}, {n,}, {n,m}, {n}?, {n,}? and {n,m}?
    // Assumes that '{' is the current character.
    // Returns either an error or moves the parser to the final '}' character.
    // (Or the '?' character if not greedy.)
    fn parse_counted(&mut self) -> Result<(), Error> {
        // Scan until the closing '}' and grab the stuff in {}.
        let start = self.chari;
        let closer =
            match self.pos('}') {
                Some(i) => i,
                None => {
                    return self.err(format!("No closing brace for counted \
                                             repetition starting at position \
                                             {}.",
                                            start).as_slice())
                }
            };
        self.chari = closer;
        let greed = try!(self.get_next_greedy());
        let inner = str::from_chars(
            self.chars.as_slice().slice(start + 1, closer));

        // Parse the min and max values from the regex.
        let (mut min, mut max): (uint, Option<uint>);
        if !inner.as_slice().contains(",") {
            min = try!(self.parse_uint(inner.as_slice()));
            max = Some(min);
        } else {
            let pieces: Vec<&str> = inner.as_slice().splitn(',', 1).collect();
            let (smin, smax) = (*pieces.get(0), *pieces.get(1));
            if smin.len() == 0 {
                return self.err("Max repetitions cannot be specified \
                                    without min repetitions.")
            }
            min = try!(self.parse_uint(smin));
            max =
                if smax.len() == 0 {
                    None
                } else {
                    Some(try!(self.parse_uint(smax)))
                };
        }

        // Do some bounds checking and make sure max >= min.
        if min > MAX_REPEAT {
            return self.err(format!(
                "{} exceeds maximum allowed repetitions ({})",
                min, MAX_REPEAT).as_slice());
        }
        if max.is_some() {
            let m = max.unwrap();
            if m > MAX_REPEAT {
                return self.err(format!(
                    "{} exceeds maximum allowed repetitions ({})",
                    m, MAX_REPEAT).as_slice());
            }
            if m < min {
                return self.err(format!(
                    "Max repetitions ({}) cannot be smaller than min \
                     repetitions ({}).", m, min).as_slice());
            }
        }

        // Now manipulate the AST be repeating elements.
        if max.is_none() {
            // Require N copies of what's on the stack and then repeat it.
            let ast = try!(self.pop_ast());
            for _ in iter::range(0, min) {
                self.push(ast.clone())
            }
            self.push(Rep(box ast, ZeroMore, greed));
        } else {
            // Require N copies of what's on the stack and then repeat it
            // up to M times optionally.
            let ast = try!(self.pop_ast());
            for _ in iter::range(0, min) {
                self.push(ast.clone())
            }
            if max.is_some() {
                for _ in iter::range(min, max.unwrap()) {
                    self.push(Rep(box ast.clone(), ZeroOne, greed))
                }
            }
            // It's possible that we popped something off the stack but
            // never put anything back on it. To keep things simple, add
            // a no-op expression.
            if min == 0 && (max.is_none() || max == Some(0)) {
                self.push(Nothing)
            }
        }
        Ok(())
    }

    // Parses all escape sequences.
    // Assumes that '\' is the current character.
    fn parse_escape(&mut self) -> Result<Ast, Error> {
        try!(self.noteof("an escape sequence following a '\\'"))

        let c = self.cur();
        if is_punct(c) {
            return Ok(Literal(c, FLAG_EMPTY))
        }
        match c {
            'a' => Ok(Literal('\x07', FLAG_EMPTY)),
            'f' => Ok(Literal('\x0C', FLAG_EMPTY)),
            't' => Ok(Literal('\t', FLAG_EMPTY)),
            'n' => Ok(Literal('\n', FLAG_EMPTY)),
            'r' => Ok(Literal('\r', FLAG_EMPTY)),
            'v' => Ok(Literal('\x0B', FLAG_EMPTY)),
            'A' => Ok(Begin(FLAG_EMPTY)),
            'z' => Ok(End(FLAG_EMPTY)),
            'b' => Ok(WordBoundary(FLAG_EMPTY)),
            'B' => Ok(WordBoundary(FLAG_NEGATED)),
            '0'|'1'|'2'|'3'|'4'|'5'|'6'|'7' => Ok(try!(self.parse_octal())),
            'x' => Ok(try!(self.parse_hex())),
            'p' | 'P' => Ok(try!(self.parse_unicode_name())),
            'd' | 'D' | 's' | 'S' | 'w' | 'W' => {
                let ranges = perl_unicode_class(c);
                let mut flags = self.flags & FLAG_NOCASE;
                if c.is_uppercase() { flags |= FLAG_NEGATED }
                Ok(Class(ranges, flags))
            }
            _ => {
                self.err(format!("Invalid escape sequence '\\\\{}'",
                                 c).as_slice())
            }
        }
    }

    // Parses a unicode character class name, either of the form \pF where
    // F is a one letter unicode class name or of the form \p{name} where
    // name is the unicode class name.
    // Assumes that \p or \P has been read (and 'p' or 'P' is the current
    // character).
    fn parse_unicode_name(&mut self) -> Result<Ast, Error> {
        let negated = if self.cur() == 'P' { FLAG_NEGATED } else { FLAG_EMPTY };
        let mut name: String;
        if self.peek_is(1, '{') {
            try!(self.expect('{'))
            let closer =
                match self.pos('}') {
                    Some(i) => i,
                    #[cfg(stage0)]
                    None => return self.err(format!(
                        "Missing '\\}' for unclosed '\\{' at position {}",
                        self.chari).as_slice()),
                    #[cfg(not(stage0))]
                    None => return self.err(format!(
                        "Missing '}}' for unclosed '{{' at position {}",
                        self.chari).as_slice()),
                };
            if closer - self.chari + 1 == 0 {
                return self.err("No Unicode class name found.")
            }
            name = self.slice(self.chari + 1, closer);
            self.chari = closer;
        } else {
            if self.chari + 1 >= self.chars.len() {
                return self.err("No single letter Unicode class name found.")
            }
            name = self.slice(self.chari + 1, self.chari + 2);
            self.chari += 1;
        }
        match find_class(UNICODE_CLASSES, name.as_slice()) {
            None => {
                return self.err(format!("Could not find Unicode class '{}'",
                                        name).as_slice())
            }
            Some(ranges) => {
                Ok(Class(ranges, negated | (self.flags & FLAG_NOCASE)))
            }
        }
    }

    // Parses an octal number, up to 3 digits.
    // Assumes that \n has been read, where n is the first digit.
    fn parse_octal(&mut self) -> Result<Ast, Error> {
        let start = self.chari;
        let mut end = start + 1;
        let (d2, d3) = (self.peek(1), self.peek(2));
        if d2 >= Some('0') && d2 <= Some('7') {
            try!(self.noteof("expected octal character in [0-7]"))
            end += 1;
            if d3 >= Some('0') && d3 <= Some('7') {
                try!(self.noteof("expected octal character in [0-7]"))
                end += 1;
            }
        }
        let s = self.slice(start, end);
        match num::from_str_radix::<u32>(s.as_slice(), 8) {
            Some(n) => Ok(Literal(try!(self.char_from_u32(n)), FLAG_EMPTY)),
            None => {
                self.err(format!("Could not parse '{}' as octal number.",
                                 s).as_slice())
            }
        }
    }

    // Parse a hex number. Either exactly two digits or anything in {}.
    // Assumes that \x has been read.
    fn parse_hex(&mut self) -> Result<Ast, Error> {
        if !self.peek_is(1, '{') {
            try!(self.expect('{'))
            return self.parse_hex_two()
        }
        let start = self.chari + 2;
        let closer =
            match self.pos('}') {
                #[cfg(stage0)]
                None => {
                    return self.err(format!("Missing '\\}' for unclosed \
                                             '\\{' at position {}",
                                            start).as_slice())
                }
                #[cfg(not(stage0))]
                None => {
                    return self.err(format!("Missing '}}' for unclosed \
                                             '{{' at position {}",
                                            start).as_slice())
                }
                Some(i) => i,
            };
        self.chari = closer;
        self.parse_hex_digits(self.slice(start, closer).as_slice())
    }

    // Parses a two-digit hex number.
    // Assumes that \xn has been read, where n is the first digit and is the
    // current character.
    // After return, parser will point at the second digit.
    fn parse_hex_two(&mut self) -> Result<Ast, Error> {
        let (start, end) = (self.chari, self.chari + 2);
        let bad = self.slice(start - 2, self.chars.len());
        try!(self.noteof(format!("Invalid hex escape sequence '{}'",
                                 bad).as_slice()))
        self.parse_hex_digits(self.slice(start, end).as_slice())
    }

    // Parses `s` as a hexadecimal number.
    fn parse_hex_digits(&self, s: &str) -> Result<Ast, Error> {
        match num::from_str_radix::<u32>(s, 16) {
            Some(n) => Ok(Literal(try!(self.char_from_u32(n)), FLAG_EMPTY)),
            None => {
                self.err(format!("Could not parse '{}' as hex number.",
                                 s).as_slice())
            }
        }
    }

    // Parses a named capture.
    // Assumes that '(?P<' has been consumed and that the current character
    // is '<'.
    // When done, parser will be at the closing '>' character.
    fn parse_named_capture(&mut self) -> Result<(), Error> {
        try!(self.noteof("a capture name"))
        let closer =
            match self.pos('>') {
                Some(i) => i,
                None => return self.err("Capture name must end with '>'."),
            };
        if closer - self.chari == 0 {
            return self.err("Capture names must have at least 1 character.")
        }
        let name = self.slice(self.chari, closer);
        if !name.as_slice().chars().all(is_valid_cap) {
            return self.err(
                "Capture names can only have underscores, letters and digits.")
        }
        if self.names.contains(&name) {
            return self.err(format!("Duplicate capture group name '{}'.",
                                    name).as_slice())
        }
        self.names.push(name.clone());
        self.chari = closer;
        self.caps += 1;
        self.stack.push(Paren(self.flags, self.caps, name));
        Ok(())
    }

    // Parses non-capture groups and options.
    // Assumes that '(?' has already been consumed and '?' is the current
    // character.
    fn parse_group_opts(&mut self) -> Result<(), Error> {
        if self.peek_is(1, 'P') && self.peek_is(2, '<') {
            try!(self.expect('P')) try!(self.expect('<'))
            return self.parse_named_capture()
        }
        let start = self.chari;
        let mut flags = self.flags;
        let mut sign = 1;
        let mut saw_flag = false;
        loop {
            try!(self.noteof("expected non-empty set of flags or closing ')'"))
            match self.cur() {
                'i' => { flags = flags | FLAG_NOCASE;     saw_flag = true},
                'm' => { flags = flags | FLAG_MULTI;      saw_flag = true},
                's' => { flags = flags | FLAG_DOTNL;      saw_flag = true},
                'U' => { flags = flags | FLAG_SWAP_GREED; saw_flag = true},
                '-' => {
                    if sign < 0 {
                        return self.err(format!(
                            "Cannot negate flags twice in '{}'.",
                            self.slice(start, self.chari + 1)).as_slice())
                    }
                    sign = -1;
                    saw_flag = false;
                    flags = flags ^ flags;
                }
                ':' | ')' => {
                    if sign < 0 {
                        if !saw_flag {
                            return self.err(format!(
                                "A valid flag does not follow negation in '{}'",
                                self.slice(start, self.chari + 1)).as_slice())
                        }
                        flags = flags ^ flags;
                    }
                    if self.cur() == ':' {
                        // Save the old flags with the opening paren.
                        self.stack.push(Paren(self.flags, 0, "".to_string()));
                    }
                    self.flags = flags;
                    return Ok(())
                }
                _ => return self.err(format!(
                    "Unrecognized flag '{}'.", self.cur()).as_slice()),
            }
        }
    }

    // Peeks at the next character and returns whether it's ungreedy or not.
    // If it is, then the next character is consumed.
    fn get_next_greedy(&mut self) -> Result<Greed, Error> {
        Ok(if self.peek_is(1, '?') {
            try!(self.expect('?'))
            Ungreedy
        } else {
            Greedy
        }.swap(self.flags & FLAG_SWAP_GREED > 0))
    }

    // Searches the stack (starting at the top) until it finds an expression
    // for which `pred` returns true. The index of that expression in the
    // stack is returned.
    // If there's no match, then one of two things happens depending on the
    // values of `allow_start`. When it's true, then `0` will be returned.
    // Otherwise, an error will be returned.
    // Generally, `allow_start` is only true when you're *not* expecting an
    // opening parenthesis.
    fn pos_last(&self, allow_start: bool, pred: |&BuildAst| -> bool)
               -> Result<uint, Error> {
        let from = match self.stack.iter().rev().position(pred) {
            Some(i) => i,
            None => {
                if allow_start {
                    self.stack.len()
                } else {
                    return self.err("No matching opening parenthesis.")
                }
            }
        };
        // Adjust index since 'from' is for the reversed stack.
        // Also, don't include the '(' or '|'.
        Ok(self.stack.len() - from)
    }

    // concat starts at `from` in the parser's stack and concatenates all
    // expressions up to the top of the stack. The resulting concatenation is
    // then pushed on to the stack.
    // Usually `from` corresponds to the position of an opening parenthesis,
    // a '|' (alternation) or the start of the entire expression.
    fn concat(&mut self, from: uint) -> Result<(), Error> {
        let ast = try!(self.build_from(from, concat_flatten));
        self.push(ast);
        Ok(())
    }

    // concat starts at `from` in the parser's stack and alternates all
    // expressions up to the top of the stack. The resulting alternation is
    // then pushed on to the stack.
    // Usually `from` corresponds to the position of an opening parenthesis
    // or the start of the entire expression.
    // This will also drop any opening parens or alternation bars found in
    // the intermediate AST.
    fn alternate(&mut self, mut from: uint) -> Result<(), Error> {
        // Unlike in the concatenation case, we want 'build_from' to continue
        // all the way to the opening left paren (so it will be popped off and
        // thrown away). But be careful with overflow---we can't count on the
        // open paren to be there.
        if from > 0 { from = from - 1}
        let ast = try!(self.build_from(from, |l,r| Alt(box l, box r)));
        self.push(ast);
        Ok(())
    }

    // build_from combines all AST elements starting at 'from' in the
    // parser's stack using 'mk' to combine them. If any such element is not an
    // AST then it is popped off the stack and ignored.
    fn build_from(&mut self, from: uint, mk: |Ast, Ast| -> Ast)
                 -> Result<Ast, Error> {
        if from >= self.stack.len() {
            return self.err("Empty group or alternate not allowed.")
        }

        let mut combined = try!(self.pop_ast());
        let mut i = self.stack.len();
        while i > from {
            i = i - 1;
            match self.stack.pop().unwrap() {
                Ast(x) => combined = mk(x, combined),
                _ => {},
            }
        }
        Ok(combined)
    }

    fn parse_uint(&self, s: &str) -> Result<uint, Error> {
        match from_str::<uint>(s) {
            Some(i) => Ok(i),
            None => {
                self.err(format!("Expected an unsigned integer but got '{}'.",
                                 s).as_slice())
            }
        }
    }

    fn char_from_u32(&self, n: u32) -> Result<char, Error> {
        match char::from_u32(n) {
            Some(c) => Ok(c),
            None => {
                self.err(format!("Could not decode '{}' to unicode \
                                  character.",
                                 n).as_slice())
            }
        }
    }

    fn pos(&self, c: char) -> Option<uint> {
        self.chars.iter()
            .skip(self.chari).position(|&c2| c2 == c).map(|i| self.chari + i)
    }

    fn err<T>(&self, msg: &str) -> Result<T, Error> {
        Err(Error {
            pos: self.chari,
            msg: msg.to_string(),
        })
    }

    fn peek(&self, offset: uint) -> Option<char> {
        if self.chari + offset >= self.chars.len() {
            return None
        }
        Some(*self.chars.get(self.chari + offset))
    }

    fn peek_is(&self, offset: uint, is: char) -> bool {
        self.peek(offset) == Some(is)
    }

    fn cur(&self) -> char {
        *self.chars.get(self.chari)
    }

    fn slice(&self, start: uint, end: uint) -> String {
        str::from_chars(self.chars.as_slice().slice(start, end)).to_string()
    }
}

// Given an unordered collection of character ranges, combine_ranges returns
// an ordered sequence of character ranges where no two ranges overlap. They
// are ordered from least to greatest (using start position).
fn combine_ranges(unordered: Vec<(char, char)>) -> Vec<(char, char)> {
    // Returns true iff the two character classes overlap or share a boundary.
    // e.g., ('a', 'g') and ('h', 'm') would return true.
    fn should_merge((a, b): (char, char), (x, y): (char, char)) -> bool {
        cmp::max(a, x) as u32 <= cmp::min(b, y) as u32 + 1
    }

    // This is currently O(n^2), but I think with sufficient cleverness,
    // it can be reduced to O(n) **if necessary**.
    let mut ordered: Vec<(char, char)> = Vec::with_capacity(unordered.len());
    for (us, ue) in unordered.move_iter() {
        let (mut us, mut ue) = (us, ue);
        assert!(us <= ue);
        let mut which: Option<uint> = None;
        for (i, &(os, oe)) in ordered.iter().enumerate() {
            if should_merge((us, ue), (os, oe)) {
                us = cmp::min(us, os);
                ue = cmp::max(ue, oe);
                which = Some(i);
                break
            }
        }
        match which {
            None => ordered.push((us, ue)),
            Some(i) => *ordered.get_mut(i) = (us, ue),
        }
    }
    ordered.sort();
    ordered
}

// Constructs a Unicode friendly Perl character class from \d, \s or \w
// (or any of their negated forms). Note that this does not handle negation.
fn perl_unicode_class(which: char) -> Vec<(char, char)> {
    match which.to_lowercase() {
        'd' => Vec::from_slice(PERLD),
        's' => Vec::from_slice(PERLS),
        'w' => Vec::from_slice(PERLW),
        _ => unreachable!(),
    }
}

// Returns a concatenation of two expressions. This also guarantees that a
// `Cat` expression will never be a direct child of another `Cat` expression.
fn concat_flatten(x: Ast, y: Ast) -> Ast {
    match (x, y) {
        (Cat(mut xs), Cat(ys)) => { xs.push_all_move(ys); Cat(xs) }
        (Cat(mut xs), ast) => { xs.push(ast); Cat(xs) }
        (ast, Cat(mut xs)) => { xs.unshift(ast); Cat(xs) }
        (ast1, ast2) => Cat(vec!(ast1, ast2)),
    }
}

pub fn is_punct(c: char) -> bool {
    match c {
        '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' |
        '[' | ']' | '{' | '}' | '^' | '$' => true,
        _ => false,
    }
}

fn is_valid_cap(c: char) -> bool {
    c == '_' || (c >= '0' && c <= '9')
    || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

fn find_class(classes: NamedClasses, name: &str) -> Option<Vec<(char, char)>> {
    match classes.bsearch(|&(s, _)| s.cmp(&name)) {
        Some(i) => Some(Vec::from_slice(classes[i].val1())),
        None => None,
    }
}

type Class = &'static [(char, char)];
type NamedClasses = &'static [(&'static str, Class)];

static ASCII_CLASSES: NamedClasses = &[
    // Classes must be in alphabetical order so that bsearch works.
    // [:alnum:]      alphanumeric (== [0-9A-Za-z])
    // [:alpha:]      alphabetic (== [A-Za-z])
    // [:ascii:]      ASCII (== [\x00-\x7F])
    // [:blank:]      blank (== [\t ])
    // [:cntrl:]      control (== [\x00-\x1F\x7F])
    // [:digit:]      digits (== [0-9])
    // [:graph:]      graphical (== [!-~])
    // [:lower:]      lower case (== [a-z])
    // [:print:]      printable (== [ -~] == [ [:graph:]])
    // [:punct:]      punctuation (== [!-/:-@[-`{-~])
    // [:space:]      whitespace (== [\t\n\v\f\r ])
    // [:upper:]      upper case (== [A-Z])
    // [:word:]       word characters (== [0-9A-Za-z_])
    // [:xdigit:]     hex digit (== [0-9A-Fa-f])
    // Taken from: http://golang.org/pkg/regex/syntax/
    ("alnum", &[('0', '9'), ('A', 'Z'), ('a', 'z')]),
    ("alpha", &[('A', 'Z'), ('a', 'z')]),
    ("ascii", &[('\x00', '\x7F')]),
    ("blank", &[(' ', ' '), ('\t', '\t')]),
    ("cntrl", &[('\x00', '\x1F'), ('\x7F', '\x7F')]),
    ("digit", &[('0', '9')]),
    ("graph", &[('!', '~')]),
    ("lower", &[('a', 'z')]),
    ("print", &[(' ', '~')]),
    ("punct", &[('!', '/'), (':', '@'), ('[', '`'), ('{', '~')]),
    ("space", &[('\t', '\t'), ('\n', '\n'), ('\x0B', '\x0B'), ('\x0C', '\x0C'),
                ('\r', '\r'), (' ', ' ')]),
    ("upper", &[('A', 'Z')]),
    ("word", &[('0', '9'), ('A', 'Z'), ('a', 'z'), ('_', '_')]),
    ("xdigit", &[('0', '9'), ('A', 'F'), ('a', 'f')]),
];
