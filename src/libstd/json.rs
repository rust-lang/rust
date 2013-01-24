// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Rust JSON serialization library
// Copyright (c) 2011 Google Inc.
#[forbid(deprecated_mode)];
#[forbid(non_camel_case_types)];

//! json serialization

use serialize;
use sort::Sort;

use core::char;
use core::cmp::{Eq, Ord};
use core::float;
use core::io::{WriterUtil, ReaderUtil};
use core::io;
use core::prelude::*;
use core::hashmap::linear::LinearMap;
use core::str;
use core::to_str;
use core::vec;

/// Represents a json value
pub enum Json {
    Number(float),
    String(~str),
    Boolean(bool),
    List(List),
    Object(~Object),
    Null,
}

pub type List = ~[Json];
pub type Object = LinearMap<~str, Json>;

pub struct Error {
    line: uint,
    col: uint,
    msg: @~str,
}

fn escape_str(s: &str) -> ~str {
    let mut escaped = ~"\"";
    for str::chars_each(s) |c| {
        match c {
          '"' => escaped += ~"\\\"",
          '\\' => escaped += ~"\\\\",
          '\x08' => escaped += ~"\\b",
          '\x0c' => escaped += ~"\\f",
          '\n' => escaped += ~"\\n",
          '\r' => escaped += ~"\\r",
          '\t' => escaped += ~"\\t",
          _ => escaped += str::from_char(c)
        }
    };

    escaped += ~"\"";

    escaped
}

fn spaces(n: uint) -> ~str {
    let mut ss = ~"";
    for n.times { str::push_str(&mut ss, " "); }
    return ss;
}

pub struct Encoder {
    priv wr: io::Writer,
}

pub fn Encoder(wr: io::Writer) -> Encoder {
    Encoder { wr: wr }
}

pub impl Encoder: serialize::Encoder {
    fn emit_nil(&self) { self.wr.write_str("null") }

    fn emit_uint(&self, v: uint) { self.emit_float(v as float); }
    fn emit_u64(&self, v: u64) { self.emit_float(v as float); }
    fn emit_u32(&self, v: u32) { self.emit_float(v as float); }
    fn emit_u16(&self, v: u16) { self.emit_float(v as float); }
    fn emit_u8(&self, v: u8)   { self.emit_float(v as float); }

    fn emit_int(&self, v: int) { self.emit_float(v as float); }
    fn emit_i64(&self, v: i64) { self.emit_float(v as float); }
    fn emit_i32(&self, v: i32) { self.emit_float(v as float); }
    fn emit_i16(&self, v: i16) { self.emit_float(v as float); }
    fn emit_i8(&self, v: i8)   { self.emit_float(v as float); }

    fn emit_bool(&self, v: bool) {
        if v {
            self.wr.write_str("true");
        } else {
            self.wr.write_str("false");
        }
    }

    fn emit_f64(&self, v: f64) { self.emit_float(v as float); }
    fn emit_f32(&self, v: f32) { self.emit_float(v as float); }
    fn emit_float(&self, v: float) {
        self.wr.write_str(float::to_str(v, 6u));
    }

    fn emit_char(&self, v: char) { self.emit_borrowed_str(str::from_char(v)) }

    fn emit_borrowed_str(&self, v: &str) { self.wr.write_str(escape_str(v)) }
    fn emit_owned_str(&self, v: &str) { self.emit_borrowed_str(v) }
    fn emit_managed_str(&self, v: &str) { self.emit_borrowed_str(v) }

    fn emit_borrowed(&self, f: fn()) { f() }
    fn emit_owned(&self, f: fn()) { f() }
    fn emit_managed(&self, f: fn()) { f() }

    fn emit_enum(&self, name: &str, f: fn()) {
        if name != "option" { fail ~"only supports option enum" }
        f()
    }
    fn emit_enum_variant(&self, _name: &str, id: uint, _cnt: uint, f: fn()) {
        if id == 0 {
            self.emit_nil();
        } else {
            f()
        }
    }
    fn emit_enum_variant_arg(&self, _idx: uint, f: fn()) {
        f()
    }

    fn emit_borrowed_vec(&self, _len: uint, f: fn()) {
        self.wr.write_char('[');
        f();
        self.wr.write_char(']');
    }
    fn emit_owned_vec(&self, len: uint, f: fn()) {
        self.emit_borrowed_vec(len, f)
    }
    fn emit_managed_vec(&self, len: uint, f: fn()) {
        self.emit_borrowed_vec(len, f)
    }
    fn emit_vec_elt(&self, idx: uint, f: fn()) {
        if idx != 0 { self.wr.write_char(','); }
        f()
    }

    fn emit_rec(&self, f: fn()) {
        self.wr.write_char('{');
        f();
        self.wr.write_char('}');
    }
    #[cfg(stage0)]
    fn emit_struct(&self, _name: &str, f: fn()) {
        self.wr.write_char('{');
        f();
        self.wr.write_char('}');
    }
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn emit_struct(&self, _name: &str, _len: uint, f: fn()) {
        self.wr.write_char('{');
        f();
        self.wr.write_char('}');
    }
    fn emit_field(&self, name: &str, idx: uint, f: fn()) {
        if idx != 0 { self.wr.write_char(','); }
        self.wr.write_str(escape_str(name));
        self.wr.write_char(':');
        f();
    }

    fn emit_tup(&self, len: uint, f: fn()) {
        self.emit_borrowed_vec(len, f);
    }
    fn emit_tup_elt(&self, idx: uint, f: fn()) {
        self.emit_vec_elt(idx, f)
    }
}

pub struct PrettyEncoder {
    priv wr: io::Writer,
    priv mut indent: uint,
}

pub fn PrettyEncoder(wr: io::Writer) -> PrettyEncoder {
    PrettyEncoder { wr: wr, indent: 0 }
}

pub impl PrettyEncoder: serialize::Encoder {
    fn emit_nil(&self) { self.wr.write_str("null") }

    fn emit_uint(&self, v: uint) { self.emit_float(v as float); }
    fn emit_u64(&self, v: u64) { self.emit_float(v as float); }
    fn emit_u32(&self, v: u32) { self.emit_float(v as float); }
    fn emit_u16(&self, v: u16) { self.emit_float(v as float); }
    fn emit_u8(&self, v: u8)   { self.emit_float(v as float); }

    fn emit_int(&self, v: int) { self.emit_float(v as float); }
    fn emit_i64(&self, v: i64) { self.emit_float(v as float); }
    fn emit_i32(&self, v: i32) { self.emit_float(v as float); }
    fn emit_i16(&self, v: i16) { self.emit_float(v as float); }
    fn emit_i8(&self, v: i8)   { self.emit_float(v as float); }

    fn emit_bool(&self, v: bool) {
        if v {
            self.wr.write_str("true");
        } else {
            self.wr.write_str("false");
        }
    }

    fn emit_f64(&self, v: f64) { self.emit_float(v as float); }
    fn emit_f32(&self, v: f32) { self.emit_float(v as float); }
    fn emit_float(&self, v: float) {
        self.wr.write_str(float::to_str(v, 6u));
    }

    fn emit_char(&self, v: char) { self.emit_borrowed_str(str::from_char(v)) }

    fn emit_borrowed_str(&self, v: &str) { self.wr.write_str(escape_str(v)); }
    fn emit_owned_str(&self, v: &str) { self.emit_borrowed_str(v) }
    fn emit_managed_str(&self, v: &str) { self.emit_borrowed_str(v) }

    fn emit_borrowed(&self, f: fn()) { f() }
    fn emit_owned(&self, f: fn()) { f() }
    fn emit_managed(&self, f: fn()) { f() }

    fn emit_enum(&self, name: &str, f: fn()) {
        if name != "option" { fail ~"only supports option enum" }
        f()
    }
    fn emit_enum_variant(&self, _name: &str, id: uint, _cnt: uint, f: fn()) {
        if id == 0 {
            self.emit_nil();
        } else {
            f()
        }
    }
    fn emit_enum_variant_arg(&self, _idx: uint, f: fn()) {
        f()
    }

    fn emit_borrowed_vec(&self, _len: uint, f: fn()) {
        self.wr.write_char('[');
        self.indent += 2;
        f();
        self.indent -= 2;
        self.wr.write_char(']');
    }
    fn emit_owned_vec(&self, len: uint, f: fn()) {
        self.emit_borrowed_vec(len, f)
    }
    fn emit_managed_vec(&self, len: uint, f: fn()) {
        self.emit_borrowed_vec(len, f)
    }
    fn emit_vec_elt(&self, idx: uint, f: fn()) {
        if idx == 0 {
            self.wr.write_char('\n');
        } else {
            self.wr.write_str(",\n");
        }
        self.wr.write_str(spaces(self.indent));
        f()
    }

    fn emit_rec(&self, f: fn()) {
        self.wr.write_char('{');
        self.indent += 2;
        f();
        self.indent -= 2;
        self.wr.write_char('}');
    }
    #[cfg(stage0)]
    fn emit_struct(&self, _name: &str, f: fn()) {
        self.emit_rec(f)
    }
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn emit_struct(&self, _name: &str, _len: uint, f: fn()) {
        self.emit_rec(f)
    }
    fn emit_field(&self, name: &str, idx: uint, f: fn()) {
        if idx == 0 {
            self.wr.write_char('\n');
        } else {
            self.wr.write_str(",\n");
        }
        self.wr.write_str(spaces(self.indent));
        self.wr.write_str(escape_str(name));
        self.wr.write_str(": ");
        f();
    }
    fn emit_tup(&self, sz: uint, f: fn()) {
        self.emit_borrowed_vec(sz, f);
    }
    fn emit_tup_elt(&self, idx: uint, f: fn()) {
        self.emit_vec_elt(idx, f)
    }
}

pub impl<S: serialize::Encoder> Json: serialize::Encodable<S> {
    fn encode(&self, s: &S) {
        match *self {
            Number(v) => v.encode(s),
            String(ref v) => v.encode(s),
            Boolean(v) => v.encode(s),
            List(ref v) => v.encode(s),
            Object(ref v) => {
                do s.emit_rec || {
                    let mut idx = 0;
                    for v.each |key, value| {
                        do s.emit_field(*key, idx) {
                            value.encode(s);
                        }
                        idx += 1;
                    }
                }
            },
            Null => s.emit_nil(),
        }
    }
}

/// Encodes a json value into a io::writer
pub fn to_writer(wr: io::Writer, json: &Json) {
    json.encode(&Encoder(wr))
}

/// Encodes a json value into a string
pub pure fn to_str(json: &Json) -> ~str {
    unsafe {
        // ugh, should be safe
        io::with_str_writer(|wr| to_writer(wr, json))
    }
}

/// Encodes a json value into a io::writer
pub fn to_pretty_writer(wr: io::Writer, json: &Json) {
    json.encode(&PrettyEncoder(wr))
}

/// Encodes a json value into a string
pub fn to_pretty_str(json: &Json) -> ~str {
    io::with_str_writer(|wr| to_pretty_writer(wr, json))
}

pub struct Parser {
    priv rdr: io::Reader,
    priv mut ch: char,
    priv mut line: uint,
    priv mut col: uint,
}

/// Decode a json value from an io::reader
pub fn Parser(rdr: io::Reader) -> Parser {
    Parser {
        rdr: rdr,
        ch: rdr.read_char(),
        line: 1,
        col: 1,
    }
}

pub impl Parser {
    fn parse() -> Result<Json, Error> {
        match move self.parse_value() {
          Ok(move value) => {
            // Skip trailing whitespaces.
            self.parse_whitespace();
            // Make sure there is no trailing characters.
            if self.eof() {
                Ok(move value)
            } else {
                self.error(~"trailing characters")
            }
          }
          Err(move e) => Err(e)
        }
    }
}

priv impl Parser {
    fn eof() -> bool { self.ch == -1 as char }

    fn bump() {
        self.ch = self.rdr.read_char();

        if self.ch == '\n' {
            self.line += 1u;
            self.col = 1u;
        } else {
            self.col += 1u;
        }
    }

    fn next_char() -> char {
        self.bump();
        self.ch
    }

    fn error<T>(msg: ~str) -> Result<T, Error> {
        Err(Error { line: self.line, col: self.col, msg: @msg })
    }

    fn parse_value() -> Result<Json, Error> {
        self.parse_whitespace();

        if self.eof() { return self.error(~"EOF while parsing value"); }

        match self.ch {
          'n' => self.parse_ident(~"ull", Null),
          't' => self.parse_ident(~"rue", Boolean(true)),
          'f' => self.parse_ident(~"alse", Boolean(false)),
          '0' .. '9' | '-' => self.parse_number(),
          '"' =>
            match move self.parse_str() {
              Ok(move s) => Ok(String(s)),
              Err(move e) => Err(e),
            },
          '[' => self.parse_list(),
          '{' => self.parse_object(),
          _ => self.error(~"invalid syntax")
        }
    }

    fn parse_whitespace() {
        while char::is_whitespace(self.ch) { self.bump(); }
    }

    fn parse_ident(ident: &str, value: Json) -> Result<Json, Error> {
        if str::all(ident, |c| c == self.next_char()) {
            self.bump();
            Ok(move value)
        } else {
            self.error(~"invalid syntax")
        }
    }

    fn parse_number() -> Result<Json, Error> {
        let mut neg = 1f;

        if self.ch == '-' {
            self.bump();
            neg = -1f;
        }

        let mut res = match self.parse_integer() {
          Ok(res) => res,
          Err(e) => return Err(e)
        };

        if self.ch == '.' {
            match self.parse_decimal(res) {
              Ok(r) => res = r,
              Err(e) => return Err(e)
            }
        }

        if self.ch == 'e' || self.ch == 'E' {
            match self.parse_exponent(res) {
              Ok(r) => res = r,
              Err(e) => return Err(e)
            }
        }

        Ok(Number(neg * res))
    }

    fn parse_integer() -> Result<float, Error> {
        let mut res = 0f;

        match self.ch {
          '0' => {
            self.bump();

            // There can be only one leading '0'.
            match self.ch {
              '0' .. '9' => return self.error(~"invalid number"),
              _ => ()
            }
          }
          '1' .. '9' => {
            while !self.eof() {
                match self.ch {
                  '0' .. '9' => {
                    res *= 10f;
                    res += ((self.ch as int) - ('0' as int)) as float;

                    self.bump();
                  }
                  _ => break
                }
            }
          }
          _ => return self.error(~"invalid number")
        }

        Ok(res)
    }

    fn parse_decimal(res: float) -> Result<float, Error> {
        self.bump();

        // Make sure a digit follows the decimal place.
        match self.ch {
          '0' .. '9' => (),
          _ => return self.error(~"invalid number")
        }

        let mut res = res;
        let mut dec = 1f;
        while !self.eof() {
            match self.ch {
              '0' .. '9' => {
                dec /= 10f;
                res += (((self.ch as int) - ('0' as int)) as float) * dec;

                self.bump();
              }
              _ => break
            }
        }

        Ok(res)
    }

    fn parse_exponent(res: float) -> Result<float, Error> {
        self.bump();

        let mut res = res;
        let mut exp = 0u;
        let mut neg_exp = false;

        match self.ch {
          '+' => self.bump(),
          '-' => { self.bump(); neg_exp = true; }
          _ => ()
        }

        // Make sure a digit follows the exponent place.
        match self.ch {
          '0' .. '9' => (),
          _ => return self.error(~"invalid number")
        }

        while !self.eof() {
            match self.ch {
              '0' .. '9' => {
                exp *= 10u;
                exp += (self.ch as uint) - ('0' as uint);

                self.bump();
              }
              _ => break
            }
        }

        let exp = float::pow_with_uint(10u, exp);
        if neg_exp {
            res /= exp;
        } else {
            res *= exp;
        }

        Ok(res)
    }

    fn parse_str() -> Result<~str, Error> {
        let mut escape = false;
        let mut res = ~"";

        while !self.eof() {
            self.bump();

            if (escape) {
                match self.ch {
                  '"' => str::push_char(&mut res, '"'),
                  '\\' => str::push_char(&mut res, '\\'),
                  '/' => str::push_char(&mut res, '/'),
                  'b' => str::push_char(&mut res, '\x08'),
                  'f' => str::push_char(&mut res, '\x0c'),
                  'n' => str::push_char(&mut res, '\n'),
                  'r' => str::push_char(&mut res, '\r'),
                  't' => str::push_char(&mut res, '\t'),
                  'u' => {
                      // Parse \u1234.
                      let mut i = 0u;
                      let mut n = 0u;
                      while i < 4u {
                          match self.next_char() {
                            '0' .. '9' => {
                              n = n * 16u + (self.ch as uint)
                                          - ('0'     as uint);
                            },
                            'a' | 'A' => n = n * 16u + 10u,
                            'b' | 'B' => n = n * 16u + 11u,
                            'c' | 'C' => n = n * 16u + 12u,
                            'd' | 'D' => n = n * 16u + 13u,
                            'e' | 'E' => n = n * 16u + 14u,
                            'f' | 'F' => n = n * 16u + 15u,
                            _ => return self.error(
                                   ~"invalid \\u escape (unrecognized hex)")
                          }
                          i += 1u;
                      }

                      // Error out if we didn't parse 4 digits.
                      if i != 4u {
                          return self.error(
                            ~"invalid \\u escape (not four digits)");
                      }

                      str::push_char(&mut res, n as char);
                  }
                  _ => return self.error(~"invalid escape")
                }
                escape = false;
            } else if self.ch == '\\' {
                escape = true;
            } else {
                if self.ch == '"' {
                    self.bump();
                    return Ok(res);
                }
                str::push_char(&mut res, self.ch);
            }
        }

        self.error(~"EOF while parsing string")
    }

    fn parse_list() -> Result<Json, Error> {
        self.bump();
        self.parse_whitespace();

        let mut values = ~[];

        if self.ch == ']' {
            self.bump();
            return Ok(List(move values));
        }

        loop {
            match move self.parse_value() {
              Ok(move v) => values.push(move v),
              Err(move e) => return Err(e)
            }

            self.parse_whitespace();
            if self.eof() {
                return self.error(~"EOF while parsing list");
            }

            match self.ch {
              ',' => self.bump(),
              ']' => { self.bump(); return Ok(List(move values)); }
              _ => return self.error(~"expected `,` or `]`")
            }
        };
    }

    fn parse_object() -> Result<Json, Error> {
        self.bump();
        self.parse_whitespace();

        let mut values = ~LinearMap::new();

        if self.ch == '}' {
          self.bump();
          return Ok(Object(move values));
        }

        while !self.eof() {
            self.parse_whitespace();

            if self.ch != '"' {
                return self.error(~"key must be a string");
            }

            let key = match move self.parse_str() {
              Ok(move key) => key,
              Err(move e) => return Err(e)
            };

            self.parse_whitespace();

            if self.ch != ':' {
                if self.eof() { break; }
                return self.error(~"expected `:`");
            }
            self.bump();

            match move self.parse_value() {
              Ok(move value) => { values.insert(key, move value); }
              Err(move e) => return Err(e)
            }
            self.parse_whitespace();

            match self.ch {
              ',' => self.bump(),
              '}' => { self.bump(); return Ok(Object(move values)); }
              _ => {
                  if self.eof() { break; }
                  return self.error(~"expected `,` or `}`");
              }
            }
        }

        return self.error(~"EOF while parsing object");
    }
}

/// Decodes a json value from an io::reader
pub fn from_reader(rdr: io::Reader) -> Result<Json, Error> {
    Parser(rdr).parse()
}

/// Decodes a json value from a string
pub fn from_str(s: &str) -> Result<Json, Error> {
    do io::with_str_reader(s) |rdr| {
        from_reader(rdr)
    }
}

pub struct Decoder {
    priv json: Json,
    priv mut stack: ~[&Json],
}

pub fn Decoder(json: Json) -> Decoder {
    Decoder { json: move json, stack: ~[] }
}

priv impl Decoder {
    fn peek(&self) -> &self/Json {
        if self.stack.len() == 0 { self.stack.push(&self.json); }
        vec::last(self.stack)
    }

    fn pop(&self) -> &self/Json {
        if self.stack.len() == 0 { self.stack.push(&self.json); }
        self.stack.pop()
    }
}

pub impl Decoder: serialize::Decoder {
    fn read_nil(&self) -> () {
        debug!("read_nil");
        match *self.pop() {
            Null => (),
            _ => fail ~"not a null"
        }
    }

    fn read_u64(&self)  -> u64  { self.read_float() as u64 }
    fn read_u32(&self)  -> u32  { self.read_float() as u32 }
    fn read_u16(&self)  -> u16  { self.read_float() as u16 }
    fn read_u8 (&self)  -> u8   { self.read_float() as u8 }
    fn read_uint(&self) -> uint { self.read_float() as uint }

    fn read_i64(&self) -> i64 { self.read_float() as i64 }
    fn read_i32(&self) -> i32 { self.read_float() as i32 }
    fn read_i16(&self) -> i16 { self.read_float() as i16 }
    fn read_i8 (&self) -> i8  { self.read_float() as i8 }
    fn read_int(&self) -> int { self.read_float() as int }

    fn read_bool(&self) -> bool {
        debug!("read_bool");
        match *self.pop() {
            Boolean(b) => b,
            _ => fail ~"not a boolean"
        }
    }

    fn read_f64(&self) -> f64 { self.read_float() as f64 }
    fn read_f32(&self) -> f32 { self.read_float() as f32 }
    fn read_float(&self) -> float {
        debug!("read_float");
        match *self.pop() {
            Number(f) => f,
            _ => fail ~"not a number"
        }
    }

    fn read_char(&self) -> char {
        let v = str::chars(self.read_owned_str());
        if v.len() != 1 { fail ~"string must have one character" }
        v[0]
    }

    fn read_owned_str(&self) -> ~str {
        debug!("read_owned_str");
        match *self.pop() {
            String(ref s) => copy *s,
            _ => fail ~"not a string"
        }
    }

    fn read_managed_str(&self) -> @str {
        debug!("read_managed_str");
        match *self.pop() {
            String(ref s) => s.to_managed(),
            _ => fail ~"not a string"
        }
    }

    fn read_owned<T>(&self, f: fn() -> T) -> T {
        debug!("read_owned()");
        f()
    }

    fn read_managed<T>(&self, f: fn() -> T) -> T {
        debug!("read_managed()");
        f()
    }

    fn read_enum<T>(&self, name: &str, f: fn() -> T) -> T {
        debug!("read_enum(%s)", name);
        if name != ~"option" { fail ~"only supports the option enum" }
        f()
    }

    fn read_enum_variant<T>(&self, f: fn(uint) -> T) -> T {
        debug!("read_enum_variant()");
        let idx = match *self.peek() {
            Null => 0,
            _ => 1,
        };
        f(idx)
    }

    fn read_enum_variant_arg<T>(&self, idx: uint, f: fn() -> T) -> T {
        debug!("read_enum_variant_arg(idx=%u)", idx);
        if idx != 0 { fail ~"unknown index" }
        f()
    }

    fn read_owned_vec<T>(&self, f: fn(uint) -> T) -> T {
        debug!("read_owned_vec()");
        let len = match *self.peek() {
            List(list) => list.len(),
            _ => fail ~"not a list",
        };
        let res = f(len);
        self.pop();
        move res
    }

    fn read_managed_vec<T>(&self, f: fn(uint) -> T) -> T {
        debug!("read_owned_vec()");
        let len = match *self.peek() {
            List(ref list) => list.len(),
            _ => fail ~"not a list",
        };
        let res = f(len);
        self.pop();
        move res
    }

    fn read_vec_elt<T>(&self, idx: uint, f: fn() -> T) -> T {
        debug!("read_vec_elt(idx=%u)", idx);
        match *self.peek() {
            List(ref list) => {
                // FIXME(#3148)---should be inferred
                let list: &self/~[Json] = list;

                self.stack.push(&list[idx]);
                f()
            }
            _ => fail ~"not a list",
        }
    }

    fn read_rec<T>(&self, f: fn() -> T) -> T {
        debug!("read_rec()");
        let value = f();
        self.pop();
        move value
    }

    #[cfg(stage0)]
    fn read_struct<T>(&self, _name: &str, f: fn() -> T) -> T {
        debug!("read_struct()");
        let value = f();
        self.pop();
        move value
    }
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn read_struct<T>(&self, _name: &str, _len: uint, f: fn() -> T) -> T {
        debug!("read_struct()");
        let value = f();
        self.pop();
        move value
    }

    fn read_field<T>(&self, name: &str, idx: uint, f: fn() -> T) -> T {
        debug!("read_rec_field(%s, idx=%u)", name, idx);
        let top = self.peek();
        match *top {
            Object(ref obj) => {
                // FIXME(#3148) This hint should not be necessary.
                let obj: &self/~Object = obj;

                match obj.find(&name.to_owned()) {
                    None => fail fmt!("no such field: %s", name),
                    Some(json) => {
                        self.stack.push(json);
                        f()
                    }
                }
            }
            Number(_) => fail ~"num",
            String(_) => fail ~"str",
            Boolean(_) => fail ~"bool",
            List(_) => fail fmt!("list: %?", top),
            Null => fail ~"null",

            //_ => fail fmt!("not an object: %?", *top)
        }
    }

    fn read_tup<T>(&self, len: uint, f: fn() -> T) -> T {
        debug!("read_tup(len=%u)", len);
        let value = f();
        self.pop();
        move value
    }

    fn read_tup_elt<T>(&self, idx: uint, f: fn() -> T) -> T {
        debug!("read_tup_elt(idx=%u)", idx);
        match *self.peek() {
            List(ref list) => {
                // FIXME(#3148)---should be inferred
                let list: &self/~[Json] = list;
                self.stack.push(&list[idx]);
                f()
            }
            _ => fail ~"not a list"
        }
    }
}

impl Json : Eq {
    pure fn eq(&self, other: &Json) -> bool {
        match (self) {
            &Number(f0) =>
                match other { &Number(f1) => f0 == f1, _ => false },
            &String(ref s0) =>
                match other { &String(ref s1) => s0 == s1, _ => false },
            &Boolean(b0) =>
                match other { &Boolean(b1) => b0 == b1, _ => false },
            &Null =>
                match other { &Null => true, _ => false },
            &List(ref v0) =>
                match other { &List(ref v1) => v0 == v1, _ => false },
            &Object(ref d0) => {
                match other {
                    &Object(ref d1) => {
                        if d0.len() == d1.len() {
                            let mut equal = true;
                            for d0.each |k, v0| {
                                match d1.find(k) {
                                    Some(v1) if v0 == v1 => { },
                                    _ => { equal = false; break }
                                }
                            };
                            equal
                        } else {
                            false
                        }
                    }
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &Json) -> bool { !self.eq(other) }
}

/// Test if two json values are less than one another
impl Json : Ord {
    pure fn lt(&self, other: &Json) -> bool {
        match (*self) {
            Number(f0) => {
                match *other {
                    Number(f1) => f0 < f1,
                    String(_) | Boolean(_) | List(_) | Object(_) |
                    Null => true
                }
            }

            String(ref s0) => {
                match *other {
                    Number(_) => false,
                    String(ref s1) => s0 < s1,
                    Boolean(_) | List(_) | Object(_) | Null => true
                }
            }

            Boolean(b0) => {
                match *other {
                    Number(_) | String(_) => false,
                    Boolean(b1) => b0 < b1,
                    List(_) | Object(_) | Null => true
                }
            }

            List(ref l0) => {
                match *other {
                    Number(_) | String(_) | Boolean(_) => false,
                    List(ref l1) => (*l0) < (*l1),
                    Object(_) | Null => true
                }
            }

            Object(ref d0) => {
                match *other {
                    Number(_) | String(_) | Boolean(_) | List(_) => false,
                    Object(ref d1) => {
                        unsafe {
                            let mut d0_flat = ~[];
                            let mut d1_flat = ~[];

                            // FIXME #4430: this is horribly inefficient...
                            for d0.each |k, v| {
                                 d0_flat.push((@copy *k, @copy *v));
                            }
                            d0_flat.qsort();

                            for d1.each |k, v| {
                                d1_flat.push((@copy *k, @copy *v));
                            }
                            d1_flat.qsort();

                            d0_flat < d1_flat
                        }
                    }
                    Null => true
                }
            }

            Null => {
                match *other {
                    Number(_) | String(_) | Boolean(_) | List(_) |
                    Object(_) =>
                        false,
                    Null => true
                }
            }
        }
    }
    pure fn le(&self, other: &Json) -> bool { !(*other).lt(&(*self)) }
    pure fn ge(&self, other: &Json) -> bool { !(*self).lt(other) }
    pure fn gt(&self, other: &Json) -> bool { (*other).lt(&(*self))  }
}

impl Error : Eq {
    pure fn eq(&self, other: &Error) -> bool {
        (*self).line == other.line &&
        (*self).col == other.col &&
        (*self).msg == other.msg
    }
    pure fn ne(&self, other: &Error) -> bool { !(*self).eq(other) }
}

trait ToJson { fn to_json() -> Json; }

impl Json: ToJson {
    fn to_json() -> Json { copy self }
}

impl @Json: ToJson {
    fn to_json() -> Json { (*self).to_json() }
}

impl int: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl i8: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl i16: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl i32: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl i64: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl uint: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl u8: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl u16: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl u32: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl u64: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl float: ToJson {
    fn to_json() -> Json { Number(self) }
}

impl f32: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl f64: ToJson {
    fn to_json() -> Json { Number(self as float) }
}

impl (): ToJson {
    fn to_json() -> Json { Null }
}

impl bool: ToJson {
    fn to_json() -> Json { Boolean(self) }
}

impl ~str: ToJson {
    fn to_json() -> Json { String(copy self) }
}

impl @~str: ToJson {
    fn to_json() -> Json { String(copy *self) }
}

impl <A: ToJson, B: ToJson> (A, B): ToJson {
    fn to_json() -> Json {
        match self {
          (ref a, ref b) => {
            List(~[a.to_json(), b.to_json()])
          }
        }
    }
}

impl <A: ToJson, B: ToJson, C: ToJson> (A, B, C): ToJson {
    fn to_json() -> Json {
        match self {
          (ref a, ref b, ref c) => {
            List(~[a.to_json(), b.to_json(), c.to_json()])
          }
        }
    }
}

impl <A: ToJson> ~[A]: ToJson {
    fn to_json() -> Json { List(self.map(|elt| elt.to_json())) }
}

impl <A: ToJson Copy> LinearMap<~str, A>: ToJson {
    fn to_json() -> Json {
        let mut d = LinearMap::new();
        for self.each() |key, value| {
            d.insert(copy *key, value.to_json());
        }
        Object(~d)
    }
}

/*
impl <A: ToJson Copy> @std::map::HashMap<~str, A>: ToJson {
    fn to_json() -> Json {
        let mut d = LinearMap::new();
        for self.each_ref |key, value| {
            d.insert(copy *key, value.to_json());
        }
        Object(~d)
    }
}
*/

impl <A: ToJson> Option<A>: ToJson {
    fn to_json() -> Json {
        match self {
          None => Null,
          Some(ref value) => value.to_json()
        }
    }
}

impl Json: to_str::ToStr {
    pure fn to_str() -> ~str { to_str(&self) }
}

impl Error: to_str::ToStr {
    pure fn to_str() -> ~str {
        fmt!("%u:%u: %s", self.line, self.col, *self.msg)
    }
}

#[cfg(test)]
mod tests {
    use core::prelude::*;

    use json::*;

    use core::result;
    use core::hashmap::linear::LinearMap;

    fn mk_object(items: &[(~str, Json)]) -> Json {
        let mut d = ~LinearMap::new();

        for items.each |item| {
            match *item {
                (copy key, copy value) => { d.insert(key, move value); },
            }
        };

        Object(move d)
    }

    #[test]
    fn test_write_null() {
        assert to_str(&Null) == ~"null";
    }

    #[test]
    fn test_write_number() {
        assert to_str(&Number(3f)) == ~"3";
        assert to_str(&Number(3.1f)) == ~"3.1";
        assert to_str(&Number(-1.5f)) == ~"-1.5";
        assert to_str(&Number(0.5f)) == ~"0.5";
    }

    #[test]
    fn test_write_str() {
        assert to_str(&String(~"")) == ~"\"\"";
        assert to_str(&String(~"foo")) == ~"\"foo\"";
    }

    #[test]
    fn test_write_bool() {
        assert to_str(&Boolean(true)) == ~"true";
        assert to_str(&Boolean(false)) == ~"false";
    }

    #[test]
    fn test_write_list() {
        assert to_str(&List(~[])) == ~"[]";
        assert to_str(&List(~[Boolean(true)])) == ~"[true]";
        assert to_str(&List(~[
            Boolean(false),
            Null,
            List(~[String(~"foo\nbar"), Number(3.5f)])
        ])) == ~"[false,null,[\"foo\\nbar\",3.5]]";
    }

    #[test]
    fn test_write_object() {
        assert to_str(&mk_object(~[])) == ~"{}";
        assert to_str(&mk_object(~[(~"a", Boolean(true))]))
            == ~"{\"a\":true}";
        let a = mk_object(~[
            (~"a", Boolean(true)),
            (~"b", List(~[
                mk_object(~[(~"c", String(~"\x0c\r"))]),
                mk_object(~[(~"d", String(~""))])
            ]))
        ]);
        // We can't compare the strings directly because the object fields be
        // printed in a different order.
        let b = result::unwrap(from_str(to_str(&a)));
        assert a == b;
    }

    #[test]
    fn test_trailing_characters() {
        assert from_str(~"nulla") ==
            Err(Error {line: 1u, col: 5u, msg: @~"trailing characters"});
        assert from_str(~"truea") ==
            Err(Error {line: 1u, col: 5u, msg: @~"trailing characters"});
        assert from_str(~"falsea") ==
            Err(Error {line: 1u, col: 6u, msg: @~"trailing characters"});
        assert from_str(~"1a") ==
            Err(Error {line: 1u, col: 2u, msg: @~"trailing characters"});
        assert from_str(~"[]a") ==
            Err(Error {line: 1u, col: 3u, msg: @~"trailing characters"});
        assert from_str(~"{}a") ==
            Err(Error {line: 1u, col: 3u, msg: @~"trailing characters"});
    }

    #[test]
    fn test_read_identifiers() {
        assert from_str(~"n") ==
            Err(Error {line: 1u, col: 2u, msg: @~"invalid syntax"});
        assert from_str(~"nul") ==
            Err(Error {line: 1u, col: 4u, msg: @~"invalid syntax"});

        assert from_str(~"t") ==
            Err(Error {line: 1u, col: 2u, msg: @~"invalid syntax"});
        assert from_str(~"truz") ==
            Err(Error {line: 1u, col: 4u, msg: @~"invalid syntax"});

        assert from_str(~"f") ==
            Err(Error {line: 1u, col: 2u, msg: @~"invalid syntax"});
        assert from_str(~"faz") ==
            Err(Error {line: 1u, col: 3u, msg: @~"invalid syntax"});

        assert from_str(~"null") == Ok(Null);
        assert from_str(~"true") == Ok(Boolean(true));
        assert from_str(~"false") == Ok(Boolean(false));
        assert from_str(~" null ") == Ok(Null);
        assert from_str(~" true ") == Ok(Boolean(true));
        assert from_str(~" false ") == Ok(Boolean(false));
    }

    #[test]
    fn test_read_number() {
        assert from_str(~"+") ==
            Err(Error {line: 1u, col: 1u, msg: @~"invalid syntax"});
        assert from_str(~".") ==
            Err(Error {line: 1u, col: 1u, msg: @~"invalid syntax"});

        assert from_str(~"-") ==
            Err(Error {line: 1u, col: 2u, msg: @~"invalid number"});
        assert from_str(~"00") ==
            Err(Error {line: 1u, col: 2u, msg: @~"invalid number"});
        assert from_str(~"1.") ==
            Err(Error {line: 1u, col: 3u, msg: @~"invalid number"});
        assert from_str(~"1e") ==
            Err(Error {line: 1u, col: 3u, msg: @~"invalid number"});
        assert from_str(~"1e+") ==
            Err(Error {line: 1u, col: 4u, msg: @~"invalid number"});

        assert from_str(~"3") == Ok(Number(3f));
        assert from_str(~"3.1") == Ok(Number(3.1f));
        assert from_str(~"-1.2") == Ok(Number(-1.2f));
        assert from_str(~"0.4") == Ok(Number(0.4f));
        assert from_str(~"0.4e5") == Ok(Number(0.4e5f));
        assert from_str(~"0.4e+15") == Ok(Number(0.4e15f));
        assert from_str(~"0.4e-01") == Ok(Number(0.4e-01f));
        assert from_str(~" 3 ") == Ok(Number(3f));
    }

    #[test]
    fn test_read_str() {
        assert from_str(~"\"") ==
            Err(Error {line: 1u, col: 2u, msg: @~"EOF while parsing string"});
        assert from_str(~"\"lol") ==
            Err(Error {line: 1u, col: 5u, msg: @~"EOF while parsing string"});

        assert from_str(~"\"\"") == Ok(String(~""));
        assert from_str(~"\"foo\"") == Ok(String(~"foo"));
        assert from_str(~"\"\\\"\"") == Ok(String(~"\""));
        assert from_str(~"\"\\b\"") == Ok(String(~"\x08"));
        assert from_str(~"\"\\n\"") == Ok(String(~"\n"));
        assert from_str(~"\"\\r\"") == Ok(String(~"\r"));
        assert from_str(~"\"\\t\"") == Ok(String(~"\t"));
        assert from_str(~" \"foo\" ") == Ok(String(~"foo"));
    }

    #[test]
    fn test_unicode_hex_escapes_in_str() {
        assert from_str(~"\"\\u12ab\"") == Ok(String(~"\u12ab"));
        assert from_str(~"\"\\uAB12\"") == Ok(String(~"\uAB12"));
    }

    #[test]
    fn test_read_list() {
        assert from_str(~"[") ==
            Err(Error {line: 1u, col: 2u, msg: @~"EOF while parsing value"});
        assert from_str(~"[1") ==
            Err(Error {line: 1u, col: 3u, msg: @~"EOF while parsing list"});
        assert from_str(~"[1,") ==
            Err(Error {line: 1u, col: 4u, msg: @~"EOF while parsing value"});
        assert from_str(~"[1,]") ==
            Err(Error {line: 1u, col: 4u, msg: @~"invalid syntax"});
        assert from_str(~"[6 7]") ==
            Err(Error {line: 1u, col: 4u, msg: @~"expected `,` or `]`"});

        assert from_str(~"[]") == Ok(List(~[]));
        assert from_str(~"[ ]") == Ok(List(~[]));
        assert from_str(~"[true]") == Ok(List(~[Boolean(true)]));
        assert from_str(~"[ false ]") == Ok(List(~[Boolean(false)]));
        assert from_str(~"[null]") == Ok(List(~[Null]));
        assert from_str(~"[3, 1]") == Ok(List(~[Number(3f), Number(1f)]));
        assert from_str(~"\n[3, 2]\n") == Ok(List(~[Number(3f), Number(2f)]));
        assert from_str(~"[2, [4, 1]]") ==
               Ok(List(~[Number(2f), List(~[Number(4f), Number(1f)])]));
    }

    #[test]
    fn test_read_object() {
        assert from_str(~"{") ==
            Err(Error {line: 1u, col: 2u, msg: @~"EOF while parsing object"});
        assert from_str(~"{ ") ==
            Err(Error {line: 1u, col: 3u, msg: @~"EOF while parsing object"});
        assert from_str(~"{1") ==
            Err(Error {line: 1u, col: 2u, msg: @~"key must be a string"});
        assert from_str(~"{ \"a\"") ==
            Err(Error {line: 1u, col: 6u, msg: @~"EOF while parsing object"});
        assert from_str(~"{\"a\"") ==
            Err(Error {line: 1u, col: 5u, msg: @~"EOF while parsing object"});
        assert from_str(~"{\"a\" ") ==
            Err(Error {line: 1u, col: 6u, msg: @~"EOF while parsing object"});

        assert from_str(~"{\"a\" 1") ==
            Err(Error {line: 1u, col: 6u, msg: @~"expected `:`"});
        assert from_str(~"{\"a\":") ==
            Err(Error {line: 1u, col: 6u, msg: @~"EOF while parsing value"});
        assert from_str(~"{\"a\":1") ==
            Err(Error {line: 1u, col: 7u, msg: @~"EOF while parsing object"});
        assert from_str(~"{\"a\":1 1") ==
            Err(Error {line: 1u, col: 8u, msg: @~"expected `,` or `}`"});
        assert from_str(~"{\"a\":1,") ==
            Err(Error {line: 1u, col: 8u, msg: @~"EOF while parsing object"});

        assert result::unwrap(from_str(~"{}")) == mk_object(~[]);
        assert result::unwrap(from_str(~"{\"a\": 3}")) ==
                  mk_object(~[(~"a", Number(3.0f))]);

        assert result::unwrap(from_str(~"{ \"a\": null, \"b\" : true }")) ==
                  mk_object(~[
                      (~"a", Null),
                      (~"b", Boolean(true))]);
        assert result::unwrap(
                      from_str(~"\n{ \"a\": null, \"b\" : true }\n")) ==
                  mk_object(~[
                      (~"a", Null),
                      (~"b", Boolean(true))]);
        assert result::unwrap(from_str(~"{\"a\" : 1.0 ,\"b\": [ true ]}")) ==
                  mk_object(~[
                      (~"a", Number(1.0)),
                      (~"b", List(~[Boolean(true)]))
                  ]);
        assert result::unwrap(from_str(
                      ~"{" +
                          ~"\"a\": 1.0, " +
                          ~"\"b\": [" +
                              ~"true," +
                              ~"\"foo\\nbar\", " +
                              ~"{ \"c\": {\"d\": null} } " +
                          ~"]" +
                      ~"}")) ==
                  mk_object(~[
                      (~"a", Number(1.0f)),
                      (~"b", List(~[
                          Boolean(true),
                          String(~"foo\nbar"),
                          mk_object(~[
                              (~"c", mk_object(~[(~"d", Null)]))
                          ])
                      ]))
                  ]);
    }

    #[test]
    fn test_multiline_errors() {
        assert from_str(~"{\n  \"foo\":\n \"bar\"") ==
            Err(Error {line: 3u, col: 8u, msg: @~"EOF while parsing object"});
    }
}
