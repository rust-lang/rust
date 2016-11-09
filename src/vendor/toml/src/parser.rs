use std::char;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::str;

macro_rules! try {
    ($e:expr) => (match $e { Some(s) => s, None => return None })
}

// We redefine Value because we need to keep track of encountered table
// definitions, eg when parsing:
//
//      [a]
//      [a.b]
//      [a]
//
// we have to error out on redefinition of [a]. This bit of data is difficult to
// track in a side table so we just have a "stripped down" AST to work with
// which has the relevant metadata fields in it.
struct TomlTable {
    values: BTreeMap<String, Value>,
    defined: bool,
}

impl TomlTable {
    fn convert(self) -> super::Table {
        self.values.into_iter().map(|(k,v)| (k, v.convert())).collect()
    }
}

enum Value {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Datetime(String),
    Array(Vec<Value>),
    Table(TomlTable),
}

impl Value {
    fn type_str(&self) -> &'static str {
        match *self {
            Value::String(..) => "string",
            Value::Integer(..) => "integer",
            Value::Float(..) => "float",
            Value::Boolean(..) => "boolean",
            Value::Datetime(..) => "datetime",
            Value::Array(..) => "array",
            Value::Table(..) => "table",
        }
    }

    fn same_type(&self, other: &Value) -> bool {
        match (self, other) {
            (&Value::String(..), &Value::String(..)) |
            (&Value::Integer(..), &Value::Integer(..)) |
            (&Value::Float(..), &Value::Float(..)) |
            (&Value::Boolean(..), &Value::Boolean(..)) |
            (&Value::Datetime(..), &Value::Datetime(..)) |
            (&Value::Array(..), &Value::Array(..)) |
            (&Value::Table(..), &Value::Table(..)) => true,

            _ => false,
        }
    }

    fn convert(self) -> super::Value {
        match self {
            Value::String(x) => super::Value::String(x),
            Value::Integer(x) => super::Value::Integer(x),
            Value::Float(x) => super::Value::Float(x),
            Value::Boolean(x) => super::Value::Boolean(x),
            Value::Datetime(x) => super::Value::Datetime(x),
            Value::Array(v) =>
                super::Value::Array(
                    v.into_iter().map(|x| x.convert()).collect()
                ),
            Value::Table(t) => super::Value::Table(t.convert())
        }
    }
}

/// Parser for converting a string to a TOML `Value` instance.
///
/// This parser contains the string slice that is being parsed, and exports the
/// list of errors which have occurred during parsing.
pub struct Parser<'a> {
    input: &'a str,
    cur: str::CharIndices<'a>,
    require_newline_after_table: bool,

    /// A list of all errors which have occurred during parsing.
    ///
    /// Not all parse errors are fatal, so this list is added to as much as
    /// possible without aborting parsing. If `None` is returned by `parse`, it
    /// is guaranteed that this list is not empty.
    pub errors: Vec<ParserError>,
}

/// A structure representing a parse error.
///
/// The data in this structure can be used to trace back to the original cause
/// of the error in order to provide diagnostics about parse errors.
#[derive(Debug, Clone)]
pub struct ParserError {
    /// The low byte at which this error is pointing at.
    pub lo: usize,
    /// One byte beyond the last character at which this error is pointing at.
    pub hi: usize,
    /// A human-readable description explaining what the error is.
    pub desc: String,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for a string.
    ///
    /// The parser can be executed by invoking the `parse` method.
    ///
    /// # Example
    ///
    /// ```
    /// let toml = r#"
    ///     [test]
    ///     foo = "bar"
    /// "#;
    ///
    /// let mut parser = toml::Parser::new(toml);
    /// match parser.parse() {
    ///     Some(value) => println!("found toml: {:?}", value),
    ///     None => {
    ///         println!("parse errors: {:?}", parser.errors);
    ///     }
    /// }
    /// ```
    pub fn new(s: &'a str) -> Parser<'a> {
        Parser {
            input: s,
            cur: s.char_indices(),
            errors: Vec::new(),
            require_newline_after_table: true,
        }
    }

    /// Converts a byte offset from an error message to a (line, column) pair
    ///
    /// All indexes are 0-based.
    pub fn to_linecol(&self, offset: usize) -> (usize, usize) {
        let mut cur = 0;
        for (i, line) in self.input.lines().enumerate() {
            if cur + line.len() + 1 > offset {
                return (i, offset - cur)
            }
            cur += line.len() + 1;
        }
        (self.input.lines().count(), 0)
    }

    /// Historical versions of toml-rs accidentally allowed a newline after a
    /// table definition, but the TOML spec requires a newline after a table
    /// definition header.
    ///
    /// This option can be set to `false` (the default is `true`) to emulate
    /// this behavior for backwards compatibility with older toml-rs versions.
    pub fn set_require_newline_after_table(&mut self, require: bool) {
        self.require_newline_after_table = require;
    }

    fn next_pos(&self) -> usize {
        self.cur.clone().next().map(|p| p.0).unwrap_or(self.input.len())
    }

    // Returns true and consumes the next character if it matches `ch`,
    // otherwise do nothing and return false
    fn eat(&mut self, ch: char) -> bool {
        match self.peek(0) {
            Some((_, c)) if c == ch => { self.cur.next(); true }
            Some(_) | None => false,
        }
    }

    // Peeks ahead `n` characters
    fn peek(&self, n: usize) -> Option<(usize, char)> {
        self.cur.clone().skip(n).next()
    }

    fn expect(&mut self, ch: char) -> bool {
        if self.eat(ch) { return true }
        let mut it = self.cur.clone();
        let lo = it.next().map(|p| p.0).unwrap_or(self.input.len());
        let hi = it.next().map(|p| p.0).unwrap_or(self.input.len());
        self.errors.push(ParserError {
            lo: lo,
            hi: hi,
            desc: match self.cur.clone().next() {
                Some((_, c)) => format!("expected `{}`, but found `{}`", ch, c),
                None => format!("expected `{}`, but found eof", ch)
            }
        });
        false
    }

    // Consumes a BOM (Byte Order Mark) if one is next
    fn bom(&mut self) -> bool {
        match self.peek(0) {
            Some((_, '\u{feff}')) => { self.cur.next(); true }
            _ => false
        }
    }

    // Consumes whitespace ('\t' and ' ') until another character (or EOF) is
    // reached. Returns if any whitespace was consumed
    fn ws(&mut self) -> bool {
        let mut ret = false;
        loop {
            match self.peek(0) {
                Some((_, '\t')) |
                Some((_, ' ')) => { self.cur.next(); ret = true; }
                _ => break,
            }
        }
        ret
    }

    // Consumes the rest of the line after a comment character
    fn comment(&mut self) -> bool {
        if !self.eat('#') { return false }
        for (_, ch) in self.cur.by_ref() {
            if ch == '\n' { break }
        }
        true
    }

    // Consumes a newline if one is next
    fn newline(&mut self) -> bool {
        match self.peek(0) {
            Some((_, '\n')) => { self.cur.next(); true }
            Some((_, '\r')) if self.peek(1).map(|c| c.1) == Some('\n') => {
                self.cur.next(); self.cur.next(); true
            }
            _ => false
        }
    }

    /// Executes the parser, parsing the string contained within.
    ///
    /// This function will return the `TomlTable` instance if parsing is
    /// successful, or it will return `None` if any parse error or invalid TOML
    /// error occurs.
    ///
    /// If an error occurs, the `errors` field of this parser can be consulted
    /// to determine the cause of the parse failure.
    pub fn parse(&mut self) -> Option<super::Table> {
        let mut ret = TomlTable { values: BTreeMap::new(), defined: false };
        self.bom();
        while self.peek(0).is_some() {
            self.ws();
            if self.newline() { continue }
            if self.comment() { continue }
            if self.eat('[') {
                let array = self.eat('[');
                let start = self.next_pos();

                // Parse the name of the section
                let mut keys = Vec::new();
                loop {
                    self.ws();
                    if let Some(s) = self.key_name() {
                        keys.push(s);
                    }
                    self.ws();
                    if self.eat(']') {
                        if array && !self.expect(']') { return None }
                        break
                    }
                    if !self.expect('.') { return None }
                }
                if keys.is_empty() { return None }

                // Build the section table
                let mut table = TomlTable {
                    values: BTreeMap::new(),
                    defined: true,
                };
                if self.require_newline_after_table {
                    self.ws();
                    if !self.comment() && !self.newline() {
                        self.errors.push(ParserError {
                            lo: start,
                            hi: start,
                            desc: format!("expected a newline after table definition"),
                        });
                        return None
                    }
                }
                if !self.values(&mut table) { return None }
                if array {
                    self.insert_array(&mut ret, &keys, Value::Table(table),
                                      start)
                } else {
                    self.insert_table(&mut ret, &keys, table, start)
                }
            } else {
                if !self.values(&mut ret) { return None }
            }
        }
        if !self.errors.is_empty() {
            None
        } else {
            Some(ret.convert())
        }
    }

    // Parse an array index as a natural number
    fn array_index(&mut self) -> Option<String> {
        self.integer(0, false, false)
    }

    /// Parse a path into a vector of paths
    pub fn lookup(&mut self) -> Option<Vec<String>> {
        if self.input.len() == 0 {
            return Some(vec![]);
        }
        let mut keys = Vec::new();
        loop {
            self.ws();
            if let Some(s) = self.key_name() {
                keys.push(s);
            } else if let Some(s) = self.array_index() {
                keys.push(s);
            } else {
                return None
            }
            self.ws();
            if !self.expect('.') { return Some(keys) }
        }
    }

    // Parse a single key name starting at `start`
    fn key_name(&mut self) -> Option<String> {
        let start = self.next_pos();
        let key = if self.eat('"') {
            self.finish_basic_string(start, false)
        } else if self.eat('\'') {
            self.finish_literal_string(start, false)
        } else {
            let mut ret = String::new();
            while let Some((_, ch)) = self.cur.clone().next() {
                match ch {
                    'a' ... 'z' |
                    'A' ... 'Z' |
                    '0' ... '9' |
                    '_' | '-' => { self.cur.next(); ret.push(ch) }
                    _ => break,
                }
            }
            Some(ret)
        };
        match key {
            Some(ref name) if name.is_empty() => {
                self.errors.push(ParserError {
                    lo: start,
                    hi: start,
                    desc: format!("expected a key but found an empty string"),
                });
                None
            }
            Some(name) => Some(name),
            None => None,
        }
    }

    // Parses the values into the given TomlTable. Returns true in case of success
    // and false in case of error.
    fn values(&mut self, into: &mut TomlTable) -> bool {
        loop {
            self.ws();
            if self.newline() { continue }
            if self.comment() { continue }
            match self.peek(0) {
                Some((_, '[')) => break,
                Some(..) => {}
                None => break,
            }
            let key_lo = self.next_pos();
            let key = match self.key_name() {
                Some(s) => s,
                None => return false
            };
            if !self.keyval_sep() { return false }
            let value = match self.value() {
                Some(value) => value,
                None => return false,
            };
            self.insert(into, key, value, key_lo);
            self.ws();
            self.comment();
            self.newline();
        }
        true
    }

    fn keyval_sep(&mut self) -> bool {
        self.ws();
        if !self.expect('=') { return false }
        self.ws();
        true
    }

    // Parses a value
    fn value(&mut self) -> Option<Value> {
        self.ws();
        match self.cur.clone().next() {
            Some((pos, '"')) => self.basic_string(pos),
            Some((pos, '\'')) => self.literal_string(pos),
            Some((pos, 't')) |
            Some((pos, 'f')) => self.boolean(pos),
            Some((pos, '[')) => self.array(pos),
            Some((pos, '{')) => self.inline_table(pos),
            Some((pos, '-')) |
            Some((pos, '+')) => self.number_or_datetime(pos),
            Some((pos, ch)) if is_digit(ch) => self.number_or_datetime(pos),
            _ => {
                let mut it = self.cur.clone();
                let lo = it.next().map(|p| p.0).unwrap_or(self.input.len());
                let hi = it.next().map(|p| p.0).unwrap_or(self.input.len());
                self.errors.push(ParserError {
                    lo: lo,
                    hi: hi,
                    desc: format!("expected a value"),
                });
                None
            }
        }
    }

    // Parses a single or multi-line string
    fn basic_string(&mut self, start: usize) -> Option<Value> {
        if !self.expect('"') { return None }
        let mut multiline = false;

        // detect multiline literals, but be careful about empty ""
        // strings
        if self.eat('"') {
            if self.eat('"') {
                multiline = true;
                self.newline();
            } else {
                // empty
                return Some(Value::String(String::new()))
            }
        }

        self.finish_basic_string(start, multiline).map(Value::String)
    }

    // Finish parsing a basic string after the opening quote has been seen
    fn finish_basic_string(&mut self,
                           start: usize,
                           multiline: bool) -> Option<String> {
        let mut ret = String::new();
        loop {
            while multiline && self.newline() { ret.push('\n') }
            match self.cur.next() {
                Some((_, '"')) => {
                    if multiline {
                        if !self.eat('"') { ret.push_str("\""); continue }
                        if !self.eat('"') { ret.push_str("\"\""); continue }
                    }
                    return Some(ret)
                }
                Some((pos, '\\')) => {
                    if let Some(c) = escape(self, pos, multiline) {
                        ret.push(c);
                    }
                }
                Some((pos, ch)) if ch < '\u{1f}' => {
                    self.errors.push(ParserError {
                        lo: pos,
                        hi: pos + 1,
                        desc: format!("control character `{}` must be escaped",
                                      ch.escape_default().collect::<String>())
                    });
                }
                Some((_, ch)) => ret.push(ch),
                None => {
                    self.errors.push(ParserError {
                        lo: start,
                        hi: self.input.len(),
                        desc: format!("unterminated string literal"),
                    });
                    return None
                }
            }
        }

        fn escape(me: &mut Parser, pos: usize, multiline: bool) -> Option<char> {
            if multiline && me.newline() {
                while me.ws() || me.newline() { /* ... */ }
                return None
            }
            match me.cur.next() {
                Some((_, 'b')) => Some('\u{8}'),
                Some((_, 't')) => Some('\u{9}'),
                Some((_, 'n')) => Some('\u{a}'),
                Some((_, 'f')) => Some('\u{c}'),
                Some((_, 'r')) => Some('\u{d}'),
                Some((_, '"')) => Some('\u{22}'),
                Some((_, '\\')) => Some('\u{5c}'),
                Some((pos, c @ 'u')) |
                Some((pos, c @ 'U')) => {
                    let len = if c == 'u' {4} else {8};
                    let num = &me.input[pos+1..];
                    let num = if num.char_indices().nth(len).map(|(i, _)| i).unwrap_or(0) == len {
                        &num[..len]
                    } else {
                        "invalid"
                    };
                    if let Some(n) = u32::from_str_radix(num, 16).ok() {
                        if let Some(c) = char::from_u32(n) {
                            me.cur.by_ref().skip(len - 1).next();
                            return Some(c)
                        } else {
                            me.errors.push(ParserError {
                                lo: pos + 1,
                                hi: pos + 5,
                                desc: format!("codepoint `{:x}` is \
                                               not a valid unicode \
                                               codepoint", n),
                            })
                        }
                    } else {
                        me.errors.push(ParserError {
                            lo: pos,
                            hi: pos + 1,
                            desc: format!("expected {} hex digits \
                                           after a `{}` escape", len, c),
                        })
                    }
                    None
                }
                Some((pos, ch)) => {
                    let next_pos = me.next_pos();
                    me.errors.push(ParserError {
                        lo: pos,
                        hi: next_pos,
                        desc: format!("unknown string escape: `{}`",
                                      ch.escape_default().collect::<String>()),
                    });
                    None
                }
                None => {
                    me.errors.push(ParserError {
                        lo: pos,
                        hi: pos + 1,
                        desc: format!("unterminated escape sequence"),
                    });
                    None
                }
            }
        }
    }

    fn literal_string(&mut self, start: usize) -> Option<Value> {
        if !self.expect('\'') { return None }
        let mut multiline = false;

        // detect multiline literals
        if self.eat('\'') {
            if self.eat('\'') {
                multiline = true;
                self.newline();
            } else {
                return Some(Value::String(String::new())) // empty
            }
        }

        self.finish_literal_string(start, multiline).map(Value::String)
    }

    fn finish_literal_string(&mut self, start: usize, multiline: bool)
                             -> Option<String> {
        let mut ret = String::new();
        loop {
            if !multiline && self.newline() {
                let next = self.next_pos();
                self.errors.push(ParserError {
                    lo: start,
                    hi: next,
                    desc: format!("literal strings cannot contain newlines"),
                });
                return None
            }
            match self.cur.next() {
                Some((_, '\'')) => {
                    if multiline {
                        if !self.eat('\'') { ret.push_str("'"); continue }
                        if !self.eat('\'') { ret.push_str("''"); continue }
                    }
                    return Some(ret)
                }
                Some((_, ch)) => ret.push(ch),
                None => {
                    self.errors.push(ParserError {
                        lo: start,
                        hi: self.input.len(),
                        desc: format!("unterminated string literal"),
                    });
                    return None
                }
            }
        }
    }

    fn number_or_datetime(&mut self, start: usize) -> Option<Value> {
        let mut is_float = false;
        let prefix = try!(self.integer(start, false, true));
        let decimal = if self.eat('.') {
            is_float = true;
            Some(try!(self.integer(start, true, false)))
        } else {
            None
        };
        let exponent = if self.eat('e') || self.eat('E') {
            is_float = true;
            Some(try!(self.integer(start, false, true)))
        } else {
            None
        };
        let end = self.next_pos();
        let input = &self.input[start..end];
        let ret = if decimal.is_none() &&
                     exponent.is_none() &&
                     !input.starts_with("+") &&
                     !input.starts_with("-") &&
                     start + 4 == end &&
                     self.eat('-') {
            self.datetime(start)
        } else {
            let input = match (decimal, exponent) {
                (None, None) => prefix,
                (Some(ref d), None) => prefix + "." + d,
                (None, Some(ref e)) => prefix + "E" + e,
                (Some(ref d), Some(ref e)) => prefix + "." + d + "E" + e,
            };
            let input = input.trim_left_matches('+');
            if is_float {
                input.parse().ok().map(Value::Float)
            } else {
                input.parse().ok().map(Value::Integer)
            }
        };
        if ret.is_none() {
            self.errors.push(ParserError {
                lo: start,
                hi: end,
                desc: format!("invalid numeric literal"),
            });
        }
        ret
    }

    fn integer(&mut self,
               start: usize,
               allow_leading_zeros: bool,
               allow_sign: bool) -> Option<String> {
        let mut s = String::new();
        if allow_sign {
            if self.eat('-') { s.push('-'); }
            else if self.eat('+') { s.push('+'); }
        }
        match self.cur.next() {
            Some((_, '0')) if !allow_leading_zeros => {
                s.push('0');
                match self.peek(0) {
                    Some((pos, c)) if '0' <= c && c <= '9' => {
                        self.errors.push(ParserError {
                            lo: start,
                            hi: pos,
                            desc: format!("leading zeroes are not allowed"),
                        });
                        return None
                    }
                    _ => {}
                }
            }
            Some((_, ch)) if '0' <= ch && ch <= '9' => {
                s.push(ch);
            }
            _ => {
                let pos = self.next_pos();
                self.errors.push(ParserError {
                    lo: pos,
                    hi: pos,
                    desc: format!("expected start of a numeric literal"),
                });
                return None;
            }
        }
        let mut underscore = false;
        loop {
            match self.cur.clone().next() {
                Some((_, ch)) if '0' <= ch && ch <= '9' => {
                    s.push(ch);
                    self.cur.next();
                    underscore = false;
                }
                Some((_, '_')) if !underscore => {
                    self.cur.next();
                    underscore = true;
                }
                Some(_) | None => break,
            }
        }
        if underscore {
            let pos = self.next_pos();
            self.errors.push(ParserError {
                lo: pos,
                hi: pos,
                desc: format!("numeral cannot end with an underscore"),
            });
            None
        } else {
            Some(s)
        }
    }

    fn boolean(&mut self, start: usize) -> Option<Value> {
        let rest = &self.input[start..];
        if rest.starts_with("true") {
            for _ in 0..4 {
                self.cur.next();
            }
            Some(Value::Boolean(true))
        } else if rest.starts_with("false") {
            for _ in 0..5 {
                self.cur.next();
            }
            Some(Value::Boolean(false))
        } else {
            let next = self.next_pos();
            self.errors.push(ParserError {
                lo: start,
                hi: next,
                desc: format!("unexpected character: `{}`",
                              rest.chars().next().unwrap()),
            });
            None
        }
    }

    fn datetime(&mut self, start: usize) -> Option<Value> {
        // Up to `start` already contains the year, and we've eaten the next
        // `-`, so we just resume parsing from there.

        let mut valid = true;

        // month
        valid = valid && digit(self.cur.next());
        valid = valid && digit(self.cur.next());

        // day
        valid = valid && self.cur.next().map(|c| c.1) == Some('-');
        valid = valid && digit(self.cur.next());
        valid = valid && digit(self.cur.next());

        valid = valid && self.cur.next().map(|c| c.1) == Some('T');

        // hour
        valid = valid && digit(self.cur.next());
        valid = valid && digit(self.cur.next());

        // minute
        valid = valid && self.cur.next().map(|c| c.1) == Some(':');
        valid = valid && digit(self.cur.next());
        valid = valid && digit(self.cur.next());

        // second
        valid = valid && self.cur.next().map(|c| c.1) == Some(':');
        valid = valid && digit(self.cur.next());
        valid = valid && digit(self.cur.next());

        // fractional seconds
        if self.eat('.') {
            valid = valid && digit(self.cur.next());
            loop {
                match self.cur.clone().next() {
                    Some((_, c)) if is_digit(c) => {
                        self.cur.next();
                    }
                    _ => break,
                }
            }
        }

        // time zone
        if !self.eat('Z') {
            valid = valid && (self.eat('+') || self.eat('-'));

            // hour
            valid = valid && digit(self.cur.next());
            valid = valid && digit(self.cur.next());

            // minute
            valid = valid && self.cur.next().map(|c| c.1) == Some(':');
            valid = valid && digit(self.cur.next());
            valid = valid && digit(self.cur.next());
        }

        return if valid {
            Some(Value::Datetime(self.input[start..self.next_pos()].to_string()))
        } else {
            let next = self.next_pos();
            self.errors.push(ParserError {
                lo: start,
                hi: start + next,
                desc: format!("malformed date literal"),
            });
            None
        };

        fn digit(val: Option<(usize, char)>) -> bool {
            match val {
                Some((_, c)) => is_digit(c),
                None => false,
            }
        }
    }

    fn array(&mut self, _start: usize) -> Option<Value> {
        if !self.expect('[') { return None }
        let mut ret = Vec::new();
        fn consume(me: &mut Parser) {
            loop {
                me.ws();
                if !me.newline() && !me.comment() { break }
            }
        }
        let mut type_str = None;
        loop {
            // Break out early if we see the closing bracket
            consume(self);
            if self.eat(']') { return Some(Value::Array(ret)) }

            // Attempt to parse a value, triggering an error if it's the wrong
            // type.
            let start = self.next_pos();
            let value = try!(self.value());
            let end = self.next_pos();
            let expected = type_str.unwrap_or(value.type_str());
            if value.type_str() != expected {
                self.errors.push(ParserError {
                    lo: start,
                    hi: end,
                    desc: format!("expected type `{}`, found type `{}`",
                                  expected, value.type_str()),
                });
            } else {
                type_str = Some(expected);
                ret.push(value);
            }

            // Look for a comma. If we don't find one we're done
            consume(self);
            if !self.eat(',') { break }
        }
        consume(self);
        if !self.expect(']') { return None }
        Some(Value::Array(ret))
    }

    fn inline_table(&mut self, _start: usize) -> Option<Value> {
        if !self.expect('{') { return None }
        self.ws();
        let mut ret = TomlTable { values: BTreeMap::new(), defined: true };
        if self.eat('}') { return Some(Value::Table(ret)) }
        loop {
            let lo = self.next_pos();
            let key = try!(self.key_name());
            if !self.keyval_sep() { return None }
            let value = try!(self.value());
            self.insert(&mut ret, key, value, lo);

            self.ws();
            if self.eat('}') { break }
            if !self.expect(',') { return None }
            self.ws();
        }
        Some(Value::Table(ret))
    }

    fn insert(&mut self, into: &mut TomlTable, key: String, value: Value,
              key_lo: usize) {
        if into.values.contains_key(&key) {
            self.errors.push(ParserError {
                lo: key_lo,
                hi: key_lo + key.len(),
                desc: format!("duplicate key: `{}`", key),
            })
        } else {
            into.values.insert(key, value);
        }
    }

    fn recurse<'b>(&mut self, mut cur: &'b mut TomlTable, keys: &'b [String],
                   key_lo: usize) -> Option<(&'b mut TomlTable, &'b str)> {
        let key_hi = keys.iter().fold(0, |a, b| a + b.len());
        for part in keys[..keys.len() - 1].iter() {
            let tmp = cur;

            if tmp.values.contains_key(part) {
                match *tmp.values.get_mut(part).unwrap() {
                    Value::Table(ref mut table) => cur = table,
                    Value::Array(ref mut array) => {
                        match array.last_mut() {
                            Some(&mut Value::Table(ref mut table)) => cur = table,
                            _ => {
                                self.errors.push(ParserError {
                                    lo: key_lo,
                                    hi: key_hi,
                                    desc: format!("array `{}` does not contain \
                                                   tables", part)
                                });
                                return None
                            }
                        }
                    }
                    _ => {
                        self.errors.push(ParserError {
                            lo: key_lo,
                            hi: key_hi,
                            desc: format!("key `{}` was not previously a table",
                                          part)
                        });
                        return None
                    }
                }
                continue
            }

            // Initialize an empty table as part of this sub-key
            tmp.values.insert(part.clone(), Value::Table(TomlTable {
                values: BTreeMap::new(),
                defined: false,
            }));
            match *tmp.values.get_mut(part).unwrap() {
                Value::Table(ref mut inner) => cur = inner,
                _ => unreachable!(),
            }
        }
        Some((cur, &**keys.last().unwrap()))
    }

    fn insert_table(&mut self, into: &mut TomlTable, keys: &[String],
                    table: TomlTable, key_lo: usize) {
        let (into, key) = match self.recurse(into, keys, key_lo) {
            Some(pair) => pair,
            None => return,
        };
        if !into.values.contains_key(key) {
            into.values.insert(key.to_owned(), Value::Table(table));
            return
        }
        if let Value::Table(ref mut into) = *into.values.get_mut(key).unwrap() {
            if into.defined {
                self.errors.push(ParserError {
                    lo: key_lo,
                    hi: key_lo + key.len(),
                    desc: format!("redefinition of table `{}`", key),
                });
            }
            for (k, v) in table.values {
                if into.values.insert(k.clone(), v).is_some() {
                    self.errors.push(ParserError {
                        lo: key_lo,
                        hi: key_lo + key.len(),
                        desc: format!("duplicate key `{}` in table", k),
                    });
                }
            }
        } else {
            self.errors.push(ParserError {
                lo: key_lo,
                hi: key_lo + key.len(),
                desc: format!("duplicate key `{}` in table", key),
            });
        }
    }

    fn insert_array(&mut self, into: &mut TomlTable,
                    keys: &[String], value: Value, key_lo: usize) {
        let (into, key) = match self.recurse(into, keys, key_lo) {
            Some(pair) => pair,
            None => return,
        };
        if !into.values.contains_key(key) {
            into.values.insert(key.to_owned(), Value::Array(Vec::new()));
        }
        match *into.values.get_mut(key).unwrap() {
            Value::Array(ref mut vec) => {
                match vec.first() {
                    Some(ref v) if !v.same_type(&value) => {
                        self.errors.push(ParserError {
                            lo: key_lo,
                            hi: key_lo + key.len(),
                            desc: format!("expected type `{}`, found type `{}`",
                                          v.type_str(), value.type_str()),
                        })
                    }
                    Some(..) | None => {}
                }
                vec.push(value);
            }
            _ => {
                self.errors.push(ParserError {
                    lo: key_lo,
                    hi: key_lo + key.len(),
                    desc: format!("key `{}` was previously not an array", key),
                });
            }
        }
    }
}

impl Error for ParserError {
    fn description(&self) -> &str { "TOML parse error" }
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.desc.fmt(f)
    }
}

fn is_digit(c: char) -> bool {
    match c { '0' ... '9' => true, _ => false }
}

#[cfg(test)]
mod tests {
    use Value::Table;
    use Parser;

    macro_rules! bad {
        ($s:expr, $msg:expr) => ({
            let mut p = Parser::new($s);
            assert!(p.parse().is_none());
            assert!(p.errors.iter().any(|e| e.desc.contains($msg)),
                    "errors: {:?}", p.errors);
        })
    }

    #[test]
    fn lookup_internal() {
        let mut parser = Parser::new(r#"hello."world\t".a.0.'escaped'.value"#);
        let result = vec![
          String::from("hello"),
          String::from("world\t"),
          String::from("a"),
          String::from("0"),
          String::from("escaped"),
          String::from("value")
        ];

        assert_eq!(parser.lookup().unwrap(), result);
    }

    #[test]
    fn lookup_internal_void() {
        let mut parser = Parser::new("");
        assert_eq!(parser.lookup().unwrap(), Vec::<String>::new());
    }

    #[test]
    fn lookup_internal_simple() {
        let mut parser = Parser::new("value");
        assert_eq!(parser.lookup().unwrap(), vec![String::from("value")]);
    }

    // This is due to key_name not parsing an empty "" correctly. Disabled for now.
    #[test]
    #[ignore]
    fn lookup_internal_quoted_void() {
        let mut parser = Parser::new("\"\"");
        assert_eq!(parser.lookup().unwrap(), vec![String::from("")]);
    }


    #[test]
    fn crlf() {
        let mut p = Parser::new("\
[project]\r\n\
\r\n\
name = \"splay\"\r\n\
version = \"0.1.0\"\r\n\
authors = [\"alex@crichton.co\"]\r\n\
\r\n\
[[lib]]\r\n\
\r\n\
path = \"lib.rs\"\r\n\
name = \"splay\"\r\n\
description = \"\"\"\
A Rust implementation of a TAR file reader and writer. This library does not\r\n\
currently handle compression, but it is abstract over all I/O readers and\r\n\
writers. Additionally, great lengths are taken to ensure that the entire\r\n\
contents are never required to be entirely resident in memory all at once.\r\n\
\"\"\"\
");
        assert!(p.parse().is_some());
    }

    #[test]
    fn linecol() {
        let p = Parser::new("ab\ncde\nf");
        assert_eq!(p.to_linecol(0), (0, 0));
        assert_eq!(p.to_linecol(1), (0, 1));
        assert_eq!(p.to_linecol(3), (1, 0));
        assert_eq!(p.to_linecol(4), (1, 1));
        assert_eq!(p.to_linecol(7), (2, 0));
    }

    #[test]
    fn fun_with_strings() {
        let mut p = Parser::new(r#"
bar = "\U00000000"
key1 = "One\nTwo"
key2 = """One\nTwo"""
key3 = """
One
Two"""

key4 = "The quick brown fox jumps over the lazy dog."
key5 = """
The quick brown \


  fox jumps over \
    the lazy dog."""
key6 = """\
       The quick brown \
       fox jumps over \
       the lazy dog.\
       """
# What you see is what you get.
winpath  = 'C:\Users\nodejs\templates'
winpath2 = '\\ServerX\admin$\system32\'
quoted   = 'Tom "Dubs" Preston-Werner'
regex    = '<\i\c*\s*>'

regex2 = '''I [dw]on't need \d{2} apples'''
lines  = '''
The first newline is
trimmed in raw strings.
   All other whitespace
   is preserved.
'''
"#);
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("bar").and_then(|k| k.as_str()), Some("\0"));
        assert_eq!(table.lookup("key1").and_then(|k| k.as_str()),
                   Some("One\nTwo"));
        assert_eq!(table.lookup("key2").and_then(|k| k.as_str()),
                   Some("One\nTwo"));
        assert_eq!(table.lookup("key3").and_then(|k| k.as_str()),
                   Some("One\nTwo"));

        let msg = "The quick brown fox jumps over the lazy dog.";
        assert_eq!(table.lookup("key4").and_then(|k| k.as_str()), Some(msg));
        assert_eq!(table.lookup("key5").and_then(|k| k.as_str()), Some(msg));
        assert_eq!(table.lookup("key6").and_then(|k| k.as_str()), Some(msg));

        assert_eq!(table.lookup("winpath").and_then(|k| k.as_str()),
                   Some(r"C:\Users\nodejs\templates"));
        assert_eq!(table.lookup("winpath2").and_then(|k| k.as_str()),
                   Some(r"\\ServerX\admin$\system32\"));
        assert_eq!(table.lookup("quoted").and_then(|k| k.as_str()),
                   Some(r#"Tom "Dubs" Preston-Werner"#));
        assert_eq!(table.lookup("regex").and_then(|k| k.as_str()),
                   Some(r"<\i\c*\s*>"));
        assert_eq!(table.lookup("regex2").and_then(|k| k.as_str()),
                   Some(r"I [dw]on't need \d{2} apples"));
        assert_eq!(table.lookup("lines").and_then(|k| k.as_str()),
                   Some("The first newline is\n\
                         trimmed in raw strings.\n   \
                            All other whitespace\n   \
                            is preserved.\n"));
    }

    #[test]
    fn tables_in_arrays() {
        let mut p = Parser::new(r#"
[[foo]]
  #…
  [foo.bar]
    #…

[[foo]] # ...
  #…
  [foo.bar]
    #...
"#);
        let table = Table(p.parse().unwrap());
        table.lookup("foo.0.bar").unwrap().as_table().unwrap();
        table.lookup("foo.1.bar").unwrap().as_table().unwrap();
    }

    #[test]
    fn fruit() {
        let mut p = Parser::new(r#"
[[fruit]]
  name = "apple"

  [fruit.physical]
    color = "red"
    shape = "round"

  [[fruit.variety]]
    name = "red delicious"

  [[fruit.variety]]
    name = "granny smith"

[[fruit]]
  name = "banana"

  [[fruit.variety]]
    name = "plantain"
"#);
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("fruit.0.name").and_then(|k| k.as_str()),
                   Some("apple"));
        assert_eq!(table.lookup("fruit.0.physical.color").and_then(|k| k.as_str()),
                   Some("red"));
        assert_eq!(table.lookup("fruit.0.physical.shape").and_then(|k| k.as_str()),
                   Some("round"));
        assert_eq!(table.lookup("fruit.0.variety.0.name").and_then(|k| k.as_str()),
                   Some("red delicious"));
        assert_eq!(table.lookup("fruit.0.variety.1.name").and_then(|k| k.as_str()),
                   Some("granny smith"));
        assert_eq!(table.lookup("fruit.1.name").and_then(|k| k.as_str()),
                   Some("banana"));
        assert_eq!(table.lookup("fruit.1.variety.0.name").and_then(|k| k.as_str()),
                   Some("plantain"));
    }

    #[test]
    fn stray_cr() {
        assert!(Parser::new("\r").parse().is_none());
        assert!(Parser::new("a = [ \r ]").parse().is_none());
        assert!(Parser::new("a = \"\"\"\r\"\"\"").parse().is_none());
        assert!(Parser::new("a = \"\"\"\\  \r  \"\"\"").parse().is_none());

        let mut p = Parser::new("foo = '''\r'''");
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("foo").and_then(|k| k.as_str()), Some("\r"));

        let mut p = Parser::new("foo = '\r'");
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("foo").and_then(|k| k.as_str()), Some("\r"));
    }

    #[test]
    fn blank_literal_string() {
        let mut p = Parser::new("foo = ''");
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("foo").and_then(|k| k.as_str()), Some(""));
    }

    #[test]
    fn many_blank() {
        let mut p = Parser::new("foo = \"\"\"\n\n\n\"\"\"");
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("foo").and_then(|k| k.as_str()), Some("\n\n"));
    }

    #[test]
    fn literal_eats_crlf() {
        let mut p = Parser::new("
            foo = \"\"\"\\\r\n\"\"\"
            bar = \"\"\"\\\r\n   \r\n   \r\n   a\"\"\"
        ");
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("foo").and_then(|k| k.as_str()), Some(""));
        assert_eq!(table.lookup("bar").and_then(|k| k.as_str()), Some("a"));
    }

    #[test]
    fn string_no_newline() {
        assert!(Parser::new("a = \"\n\"").parse().is_none());
        assert!(Parser::new("a = '\n'").parse().is_none());
    }

    #[test]
    fn bad_leading_zeros() {
        assert!(Parser::new("a = 00").parse().is_none());
        assert!(Parser::new("a = -00").parse().is_none());
        assert!(Parser::new("a = +00").parse().is_none());
        assert!(Parser::new("a = 00.0").parse().is_none());
        assert!(Parser::new("a = -00.0").parse().is_none());
        assert!(Parser::new("a = +00.0").parse().is_none());
        assert!(Parser::new("a = 9223372036854775808").parse().is_none());
        assert!(Parser::new("a = -9223372036854775809").parse().is_none());
    }

    #[test]
    fn bad_floats() {
        assert!(Parser::new("a = 0.").parse().is_none());
        assert!(Parser::new("a = 0.e").parse().is_none());
        assert!(Parser::new("a = 0.E").parse().is_none());
        assert!(Parser::new("a = 0.0E").parse().is_none());
        assert!(Parser::new("a = 0.0e").parse().is_none());
        assert!(Parser::new("a = 0.0e-").parse().is_none());
        assert!(Parser::new("a = 0.0e+").parse().is_none());
        assert!(Parser::new("a = 0.0e+00").parse().is_none());
    }

    #[test]
    fn floats() {
        macro_rules! t {
            ($actual:expr, $expected:expr) => ({
                let f = format!("foo = {}", $actual);
                let mut p = Parser::new(&f);
                let table = Table(p.parse().unwrap());
                assert_eq!(table.lookup("foo").and_then(|k| k.as_float()),
                           Some($expected));
            })
        }

        t!("1.0", 1.0);
        t!("1.0e0", 1.0);
        t!("1.0e+0", 1.0);
        t!("1.0e-0", 1.0);
        t!("1.001e-0", 1.001);
        t!("2e10", 2e10);
        t!("2e+10", 2e10);
        t!("2e-10", 2e-10);
        t!("2_0.0", 20.0);
        t!("2_0.0_0e0_0", 20.0);
        t!("2_0.1_0e1_0", 20.1e10);
    }

    #[test]
    fn bare_key_names() {
        let mut p = Parser::new("
            foo = 3
            foo_3 = 3
            foo_-2--3--r23f--4-f2-4 = 3
            _ = 3
            - = 3
            8 = 8
            \"a\" = 3
            \"!\" = 3
            \"a^b\" = 3
            \"\\\"\" = 3
            \"character encoding\" = \"value\"
            'ʎǝʞ' = \"value\"
        ");
        let table = Table(p.parse().unwrap());
        assert!(table.lookup("foo").is_some());
        assert!(table.lookup("-").is_some());
        assert!(table.lookup("_").is_some());
        assert!(table.lookup("8").is_some());
        assert!(table.lookup("foo_3").is_some());
        assert!(table.lookup("foo_-2--3--r23f--4-f2-4").is_some());
        assert!(table.lookup("a").is_some());
        assert!(table.lookup("\"!\"").is_some());
        assert!(table.lookup("\"\\\"\"").is_some());
        assert!(table.lookup("\"character encoding\"").is_some());
        assert!(table.lookup("'ʎǝʞ'").is_some());
    }

    #[test]
    fn bad_keys() {
        assert!(Parser::new("key\n=3").parse().is_none());
        assert!(Parser::new("key=\n3").parse().is_none());
        assert!(Parser::new("key|=3").parse().is_none());
        assert!(Parser::new("\"\"=3").parse().is_none());
        assert!(Parser::new("=3").parse().is_none());
        assert!(Parser::new("\"\"|=3").parse().is_none());
        assert!(Parser::new("\"\n\"|=3").parse().is_none());
        assert!(Parser::new("\"\r\"|=3").parse().is_none());
    }

    #[test]
    fn bad_table_names() {
        assert!(Parser::new("[]").parse().is_none());
        assert!(Parser::new("[.]").parse().is_none());
        assert!(Parser::new("[\"\".\"\"]").parse().is_none());
        assert!(Parser::new("[a.]").parse().is_none());
        assert!(Parser::new("[\"\"]").parse().is_none());
        assert!(Parser::new("[!]").parse().is_none());
        assert!(Parser::new("[\"\n\"]").parse().is_none());
        assert!(Parser::new("[a.b]\n[a.\"b\"]").parse().is_none());
        assert!(Parser::new("[']").parse().is_none());
        assert!(Parser::new("[''']").parse().is_none());
        assert!(Parser::new("['''''']").parse().is_none());
        assert!(Parser::new("['\n']").parse().is_none());
        assert!(Parser::new("['\r\n']").parse().is_none());
    }

    #[test]
    fn table_names() {
        let mut p = Parser::new("
            [a.\"b\"]
            [\"f f\"]
            [\"f.f\"]
            [\"\\\"\"]
            ['a.a']
            ['\"\"']
        ");
        let table = Table(p.parse().unwrap());
        assert!(table.lookup("a.b").is_some());
        assert!(table.lookup("\"f f\"").is_some());
        assert!(table.lookup("\"\\\"\"").is_some());
        assert!(table.lookup("'\"\"'").is_some());
    }

    #[test]
    fn invalid_bare_numeral() {
        assert!(Parser::new("4").parse().is_none());
    }

    #[test]
    fn inline_tables() {
        assert!(Parser::new("a = {}").parse().is_some());
        assert!(Parser::new("a = {b=1}").parse().is_some());
        assert!(Parser::new("a = {   b   =   1    }").parse().is_some());
        assert!(Parser::new("a = {a=1,b=2}").parse().is_some());
        assert!(Parser::new("a = {a=1,b=2,c={}}").parse().is_some());
        assert!(Parser::new("a = {a=1,}").parse().is_none());
        assert!(Parser::new("a = {,}").parse().is_none());
        assert!(Parser::new("a = {a=1,a=1}").parse().is_none());
        assert!(Parser::new("a = {\n}").parse().is_none());
        assert!(Parser::new("a = {").parse().is_none());
        assert!(Parser::new("a = {a=[\n]}").parse().is_some());
        assert!(Parser::new("a = {\"a\"=[\n]}").parse().is_some());
        assert!(Parser::new("a = [\n{},\n{},\n]").parse().is_some());
    }

    #[test]
    fn number_underscores() {
        macro_rules! t {
            ($actual:expr, $expected:expr) => ({
                let f = format!("foo = {}", $actual);
                let mut p = Parser::new(&f);
                let table = Table(p.parse().unwrap());
                assert_eq!(table.lookup("foo").and_then(|k| k.as_integer()),
                           Some($expected));
            })
        }

        t!("1_0", 10);
        t!("1_0_0", 100);
        t!("1_000", 1000);
        t!("+1_000", 1000);
        t!("-1_000", -1000);
    }

    #[test]
    fn bad_underscores() {
        assert!(Parser::new("foo = 0_").parse().is_none());
        assert!(Parser::new("foo = 0__0").parse().is_none());
        assert!(Parser::new("foo = __0").parse().is_none());
        assert!(Parser::new("foo = 1_0_").parse().is_none());
    }

    #[test]
    fn bad_unicode_codepoint() {
        bad!("foo = \"\\uD800\"", "not a valid unicode codepoint");
    }

    #[test]
    fn bad_strings() {
        bad!("foo = \"\\uxx\"", "expected 4 hex digits");
        bad!("foo = \"\\u\"", "expected 4 hex digits");
        bad!("foo = \"\\", "unterminated");
        bad!("foo = '", "unterminated");
    }

    #[test]
    fn empty_string() {
        let mut p = Parser::new("foo = \"\"");
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("foo").unwrap().as_str(), Some(""));
    }

    #[test]
    fn booleans() {
        let mut p = Parser::new("foo = true");
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("foo").unwrap().as_bool(), Some(true));

        let mut p = Parser::new("foo = false");
        let table = Table(p.parse().unwrap());
        assert_eq!(table.lookup("foo").unwrap().as_bool(), Some(false));

        assert!(Parser::new("foo = true2").parse().is_none());
        assert!(Parser::new("foo = false2").parse().is_none());
        assert!(Parser::new("foo = t1").parse().is_none());
        assert!(Parser::new("foo = f2").parse().is_none());
    }

    #[test]
    fn bad_nesting() {
        bad!("
            a = [2]
            [[a]]
            b = 5
        ", "expected type `integer`, found type `table`");
        bad!("
            a = 1
            [a.b]
        ", "key `a` was not previously a table");
        bad!("
            a = []
            [a.b]
        ", "array `a` does not contain tables");
        bad!("
            a = []
            [[a.b]]
        ", "array `a` does not contain tables");
        bad!("
            [a]
            b = { c = 2, d = {} }
            [a.b]
            c = 2
        ", "duplicate key `c` in table");
    }

    #[test]
    fn bad_table_redefine() {
        bad!("
            [a]
            foo=\"bar\"
            [a.b]
            foo=\"bar\"
            [a]
        ", "redefinition of table `a`");
        bad!("
            [a]
            foo=\"bar\"
            b = { foo = \"bar\" }
            [a]
        ", "redefinition of table `a`");
        bad!("
            [a]
            b = {}
            [a.b]
        ", "redefinition of table `b`");

        bad!("
            [a]
            b = {}
            [a]
        ", "redefinition of table `a`");
    }

    #[test]
    fn datetimes() {
        macro_rules! t {
            ($actual:expr) => ({
                let f = format!("foo = {}", $actual);
                let mut p = Parser::new(&f);
                let table = Table(p.parse().unwrap());
                assert_eq!(table.lookup("foo").and_then(|k| k.as_datetime()),
                           Some($actual));
            })
        }

        t!("2016-09-09T09:09:09Z");
        t!("2016-09-09T09:09:09.0Z");
        t!("2016-09-09T09:09:09.0+10:00");
        t!("2016-09-09T09:09:09.01234567890-02:00");
        bad!("foo = 2016-09-09T09:09:09.Z", "malformed date literal");
        bad!("foo = 2016-9-09T09:09:09Z", "malformed date literal");
        bad!("foo = 2016-09-09T09:09:09+2:00", "malformed date literal");
        bad!("foo = 2016-09-09T09:09:09-2:00", "malformed date literal");
        bad!("foo = 2016-09-09T09:09:09Z-2:00", "expected");
    }
}
