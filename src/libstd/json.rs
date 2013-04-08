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
#[forbid(non_camel_case_types)];

//! json serialization

use core::prelude::*;
use core::io::{WriterUtil, ReaderUtil};
use core::hashmap::HashMap;

use serialize::Encodable;
use serialize;
use sort::Sort;

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
pub type Object = HashMap<~str, Json>;

#[deriving(Eq)]
pub struct Error {
    line: uint,
    col: uint,
    msg: @~str,
}

fn escape_str(s: &str) -> ~str {
    let mut escaped = ~"\"";
    for str::each_char(s) |c| {
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
    priv wr: @io::Writer,
}

pub fn Encoder(wr: @io::Writer) -> Encoder {
    Encoder { wr: wr }
}

impl serialize::Encoder for Encoder {
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
        self.wr.write_str(float::to_str_digits(v, 6u));
    }

    fn emit_char(&self, v: char) { self.emit_str(str::from_char(v)) }
    fn emit_str(&self, v: &str) { self.wr.write_str(escape_str(v)) }

    fn emit_enum(&self, _name: &str, f: &fn()) { f() }
    fn emit_enum_variant(&self, name: &str, _id: uint, cnt: uint, f: &fn()) {
        // enums are encoded as strings or vectors:
        // Bunny => "Bunny"
        // Kangaroo(34,"William") => ["Kangaroo",[34,"William"]]

        if cnt == 0 {
            self.wr.write_str(escape_str(name));
        } else {
            self.wr.write_char('[');
            self.wr.write_str(escape_str(name));
            self.wr.write_char(',');
            f();
            self.wr.write_char(']');
        }
    }

    fn emit_enum_variant_arg(&self, idx: uint, f: &fn()) {
        if (idx != 0) {self.wr.write_char(',');}
        f();
    }

    fn emit_seq(&self, _len: uint, f: &fn()) {
        self.wr.write_char('[');
        f();
        self.wr.write_char(']');
    }

    fn emit_seq_elt(&self, idx: uint, f: &fn()) {
        if idx != 0 { self.wr.write_char(','); }
        f()
    }

    fn emit_struct(&self, _name: &str, _len: uint, f: &fn()) {
        self.wr.write_char('{');
        f();
        self.wr.write_char('}');
    }
    fn emit_field(&self, name: &str, idx: uint, f: &fn()) {
        if idx != 0 { self.wr.write_char(','); }
        self.wr.write_str(escape_str(name));
        self.wr.write_char(':');
        f();
    }

    fn emit_option(&self, f: &fn()) { f(); }
    fn emit_option_none(&self) { self.emit_nil(); }
    fn emit_option_some(&self, f: &fn()) { f(); }

    fn emit_map(&self, _len: uint, f: &fn()) {
        self.wr.write_char('{');
        f();
        self.wr.write_char('}');
    }

    fn emit_map_elt_key(&self, idx: uint, f: &fn()) {
        if idx != 0 { self.wr.write_char(','); }
        f()
    }

    fn emit_map_elt_val(&self, _idx: uint, f: &fn()) {
        self.wr.write_char(':');
        f()
    }
}

pub struct PrettyEncoder {
    priv wr: @io::Writer,
    priv mut indent: uint,
}

pub fn PrettyEncoder(wr: @io::Writer) -> PrettyEncoder {
    PrettyEncoder { wr: wr, indent: 0 }
}

impl serialize::Encoder for PrettyEncoder {
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
        self.wr.write_str(float::to_str_digits(v, 6u));
    }

    fn emit_char(&self, v: char) { self.emit_str(str::from_char(v)) }
    fn emit_str(&self, v: &str) { self.wr.write_str(escape_str(v)); }

    fn emit_enum(&self, _name: &str, f: &fn()) { f() }
    fn emit_enum_variant(&self, name: &str, _id: uint, cnt: uint, f: &fn()) {
        if cnt == 0 {
            self.wr.write_str(escape_str(name));
        } else {
            self.wr.write_char('[');
            self.indent += 2;
            self.wr.write_char('\n');
            self.wr.write_str(spaces(self.indent));
            self.wr.write_str(escape_str(name));
            self.wr.write_str(",\n");
            f();
            self.wr.write_char('\n');
            self.indent -= 2;
            self.wr.write_str(spaces(self.indent));
            self.wr.write_char(']');
        }
    }
    fn emit_enum_variant_arg(&self, idx: uint, f: &fn()) {
        if idx != 0 {
            self.wr.write_str(",\n");
        }
        self.wr.write_str(spaces(self.indent));
        f()
    }

    fn emit_seq(&self, len: uint, f: &fn()) {
        if len == 0 {
            self.wr.write_str("[]");
        } else {
            self.wr.write_char('[');
            self.indent += 2;
            f();
            self.wr.write_char('\n');
            self.indent -= 2;
            self.wr.write_str(spaces(self.indent));
            self.wr.write_char(']');
        }
    }
    fn emit_seq_elt(&self, idx: uint, f: &fn()) {
        if idx == 0 {
            self.wr.write_char('\n');
        } else {
            self.wr.write_str(",\n");
        }
        self.wr.write_str(spaces(self.indent));
        f()
    }

    fn emit_struct(&self, _name: &str, len: uint, f: &fn()) {
        if len == 0 {
            self.wr.write_str("{}");
        } else {
            self.wr.write_char('{');
            self.indent += 2;
            f();
            self.wr.write_char('\n');
            self.indent -= 2;
            self.wr.write_str(spaces(self.indent));
            self.wr.write_char('}');
        }
    }
    fn emit_field(&self, name: &str, idx: uint, f: &fn()) {
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

    fn emit_option(&self, f: &fn()) { f(); }
    fn emit_option_none(&self) { self.emit_nil(); }
    fn emit_option_some(&self, f: &fn()) { f(); }

    fn emit_map(&self, len: uint, f: &fn()) {
        if len == 0 {
            self.wr.write_str("{}");
        } else {
            self.wr.write_char('{');
            self.indent += 2;
            f();
            self.wr.write_char('\n');
            self.indent -= 2;
            self.wr.write_str(spaces(self.indent));
            self.wr.write_char('}');
        }
    }
    fn emit_map_elt_key(&self, idx: uint, f: &fn()) {
        if idx == 0 {
            self.wr.write_char('\n');
        } else {
            self.wr.write_str(",\n");
        }
        self.wr.write_str(spaces(self.indent));
        f();
    }

    fn emit_map_elt_val(&self, _idx: uint, f: &fn()) {
        self.wr.write_str(": ");
        f();
    }
}

impl<E: serialize::Encoder> serialize::Encodable<E> for Json {
    fn encode(&self, e: &E) {
        match *self {
            Number(v) => v.encode(e),
            String(ref v) => v.encode(e),
            Boolean(v) => v.encode(e),
            List(ref v) => v.encode(e),
            Object(ref v) => v.encode(e),
            Null => e.emit_nil(),
        }
    }
}

/// Encodes a json value into a io::writer
pub fn to_writer(wr: @io::Writer, json: &Json) {
    json.encode(&Encoder(wr))
}

/// Encodes a json value into a string
pub fn to_str(json: &Json) -> ~str {
    io::with_str_writer(|wr| to_writer(wr, json))
}

/// Encodes a json value into a io::writer
pub fn to_pretty_writer(wr: @io::Writer, json: &Json) {
    json.encode(&PrettyEncoder(wr))
}

/// Encodes a json value into a string
pub fn to_pretty_str(json: &Json) -> ~str {
    io::with_str_writer(|wr| to_pretty_writer(wr, json))
}

pub struct Parser {
    priv rdr: @io::Reader,
    priv ch: char,
    priv line: uint,
    priv col: uint,
}

/// Decode a json value from an io::reader
pub fn Parser(rdr: @io::Reader) -> Parser {
    Parser {
        rdr: rdr,
        ch: rdr.read_char(),
        line: 1,
        col: 1,
    }
}

pub impl Parser {
    fn parse(&mut self) -> Result<Json, Error> {
        match self.parse_value() {
          Ok(value) => {
            // Skip trailing whitespaces.
            self.parse_whitespace();
            // Make sure there is no trailing characters.
            if self.eof() {
                Ok(value)
            } else {
                self.error(~"trailing characters")
            }
          }
          Err(e) => Err(e)
        }
    }
}

priv impl Parser {
    fn eof(&self) -> bool { self.ch == -1 as char }

    fn bump(&mut self) {
        self.ch = self.rdr.read_char();

        if self.ch == '\n' {
            self.line += 1u;
            self.col = 1u;
        } else {
            self.col += 1u;
        }
    }

    fn next_char(&mut self) -> char {
        self.bump();
        self.ch
    }

    fn error<T>(&self, msg: ~str) -> Result<T, Error> {
        Err(Error { line: self.line, col: self.col, msg: @msg })
    }

    fn parse_value(&mut self) -> Result<Json, Error> {
        self.parse_whitespace();

        if self.eof() { return self.error(~"EOF while parsing value"); }

        match self.ch {
          'n' => self.parse_ident(~"ull", Null),
          't' => self.parse_ident(~"rue", Boolean(true)),
          'f' => self.parse_ident(~"alse", Boolean(false)),
          '0' .. '9' | '-' => self.parse_number(),
          '"' =>
            match self.parse_str() {
              Ok(s) => Ok(String(s)),
              Err(e) => Err(e),
            },
          '[' => self.parse_list(),
          '{' => self.parse_object(),
          _ => self.error(~"invalid syntax")
        }
    }

    fn parse_whitespace(&mut self) {
        while char::is_whitespace(self.ch) { self.bump(); }
    }

    fn parse_ident(&mut self, ident: &str, value: Json) -> Result<Json, Error> {
        if str::all(ident, |c| c == self.next_char()) {
            self.bump();
            Ok(value)
        } else {
            self.error(~"invalid syntax")
        }
    }

    fn parse_number(&mut self) -> Result<Json, Error> {
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

    fn parse_integer(&mut self) -> Result<float, Error> {
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

    fn parse_decimal(&mut self, res: float) -> Result<float, Error> {
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

    fn parse_exponent(&mut self, mut res: float) -> Result<float, Error> {
        self.bump();

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

    fn parse_str(&mut self) -> Result<~str, Error> {
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

    fn parse_list(&mut self) -> Result<Json, Error> {
        self.bump();
        self.parse_whitespace();

        let mut values = ~[];

        if self.ch == ']' {
            self.bump();
            return Ok(List(values));
        }

        loop {
            match self.parse_value() {
              Ok(v) => values.push(v),
              Err(e) => return Err(e)
            }

            self.parse_whitespace();
            if self.eof() {
                return self.error(~"EOF while parsing list");
            }

            match self.ch {
              ',' => self.bump(),
              ']' => { self.bump(); return Ok(List(values)); }
              _ => return self.error(~"expected `,` or `]`")
            }
        };
    }

    fn parse_object(&mut self) -> Result<Json, Error> {
        self.bump();
        self.parse_whitespace();

        let mut values = ~HashMap::new();

        if self.ch == '}' {
          self.bump();
          return Ok(Object(values));
        }

        while !self.eof() {
            self.parse_whitespace();

            if self.ch != '"' {
                return self.error(~"key must be a string");
            }

            let key = match self.parse_str() {
              Ok(key) => key,
              Err(e) => return Err(e)
            };

            self.parse_whitespace();

            if self.ch != ':' {
                if self.eof() { break; }
                return self.error(~"expected `:`");
            }
            self.bump();

            match self.parse_value() {
              Ok(value) => { values.insert(key, value); }
              Err(e) => return Err(e)
            }
            self.parse_whitespace();

            match self.ch {
              ',' => self.bump(),
              '}' => { self.bump(); return Ok(Object(values)); }
              _ => {
                  if self.eof() { break; }
                  return self.error(~"expected `,` or `}`");
              }
            }
        }

        return self.error(~"EOF while parsing object");
    }
}

/// Decodes a json value from an @io::Reader
pub fn from_reader(rdr: @io::Reader) -> Result<Json, Error> {
    let mut parser = Parser(rdr);
    parser.parse()
}

/// Decodes a json value from a string
pub fn from_str(s: &str) -> Result<Json, Error> {
    do io::with_str_reader(s) |rdr| {
        from_reader(rdr)
    }
}

pub struct Decoder {
    priv mut stack: ~[Json],
}

pub fn Decoder(json: Json) -> Decoder {
    Decoder { stack: ~[json] }
}

impl serialize::Decoder for Decoder {
    fn read_nil(&self) -> () {
        debug!("read_nil");
        match self.stack.pop() {
            Null => (),
            value => fail!(fmt!("not a null: %?", value))
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
        match self.stack.pop() {
            Boolean(b) => b,
            value => fail!(fmt!("not a boolean: %?", value))
        }
    }

    fn read_f64(&self) -> f64 { self.read_float() as f64 }
    fn read_f32(&self) -> f32 { self.read_float() as f32 }
    fn read_float(&self) -> float {
        debug!("read_float");
        match self.stack.pop() {
            Number(f) => f,
            value => fail!(fmt!("not a number: %?", value))
        }
    }

    fn read_char(&self) -> char {
        let mut v = ~[];
        for str::each_char(self.read_str()) |c| { v.push(c) }
        if v.len() != 1 { fail!(~"string must have one character") }
        v[0]
    }

    fn read_str(&self) -> ~str {
        debug!("read_str");
        match self.stack.pop() {
            String(s) => s,
            json => fail!(fmt!("not a string: %?", json))
        }
    }

    fn read_enum<T>(&self, name: &str, f: &fn() -> T) -> T {
        debug!("read_enum(%s)", name);
        f()
    }

    fn read_enum_variant<T>(&self, names: &[&str], f: &fn(uint) -> T) -> T {
        debug!("read_enum_variant(names=%?)", names);
        let name = match self.stack.pop() {
            String(s) => s,
            List(list) => {
                do vec::consume_reverse(list) |_i, v| {
                    self.stack.push(v);
                }
                match self.stack.pop() {
                    String(s) => s,
                    value => fail!(fmt!("invalid variant name: %?", value)),
                }
            }
            ref json => fail!(fmt!("invalid variant: %?", *json)),
        };
        let idx = match vec::position(names, |n| str::eq_slice(*n, name)) {
            Some(idx) => idx,
            None => fail!(fmt!("Unknown variant name: %?", name)),
        };
        f(idx)
    }

    fn read_enum_variant_arg<T>(&self, idx: uint, f: &fn() -> T) -> T {
        debug!("read_enum_variant_arg(idx=%u)", idx);
        f()
    }

    fn read_seq<T>(&self, f: &fn(uint) -> T) -> T {
        debug!("read_seq()");
        let len = match self.stack.pop() {
            List(list) => {
                let len = list.len();
                do vec::consume_reverse(list) |_i, v| {
                    self.stack.push(v);
                }
                len
            }
            _ => fail!(~"not a list"),
        };
        f(len)
    }

    fn read_seq_elt<T>(&self, idx: uint, f: &fn() -> T) -> T {
        debug!("read_seq_elt(idx=%u)", idx);
        f()
    }

    fn read_struct<T>(&self, name: &str, len: uint, f: &fn() -> T) -> T {
        debug!("read_struct(name=%s, len=%u)", name, len);
        let value = f();
        self.stack.pop();
        value
    }

    fn read_field<T>(&self, name: &str, idx: uint, f: &fn() -> T) -> T {
        debug!("read_field(%s, idx=%u)", name, idx);
        match self.stack.pop() {
            Object(obj) => {
                let mut obj = obj;
                let value = match obj.pop(&name.to_owned()) {
                    None => fail!(fmt!("no such field: %s", name)),
                    Some(json) => {
                        self.stack.push(json);
                        f()
                    }
                };
                self.stack.push(Object(obj));
                value
            }
            value => fail!(fmt!("not an object: %?", value))
        }
    }

    fn read_option<T>(&self, f: &fn(bool) -> T) -> T {
        match self.stack.pop() {
            Null => f(false),
            value => { self.stack.push(value); f(true) }
        }
    }

    fn read_map<T>(&self, f: &fn(uint) -> T) -> T {
        debug!("read_map()");
        let len = match self.stack.pop() {
            Object(obj) => {
                let mut obj = obj;
                let len = obj.len();
                do obj.consume |key, value| {
                    self.stack.push(value);
                    self.stack.push(String(key));
                }
                len
            }
            json => fail!(fmt!("not an object: %?", json)),
        };
        f(len)
    }

    fn read_map_elt_key<T>(&self, idx: uint, f: &fn() -> T) -> T {
        debug!("read_map_elt_key(idx=%u)", idx);
        f()
    }

    fn read_map_elt_val<T>(&self, idx: uint, f: &fn() -> T) -> T {
        debug!("read_map_elt_val(idx=%u)", idx);
        f()
    }
}

impl Eq for Json {
    fn eq(&self, other: &Json) -> bool {
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
                            for d0.each |&(k, v0)| {
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
    fn ne(&self, other: &Json) -> bool { !self.eq(other) }
}

/// Test if two json values are less than one another
impl Ord for Json {
    fn lt(&self, other: &Json) -> bool {
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
                        let mut d0_flat = ~[];
                        let mut d1_flat = ~[];

                        // FIXME #4430: this is horribly inefficient...
                        for d0.each |&(k, v)| {
                             d0_flat.push((@copy *k, @copy *v));
                        }
                        d0_flat.qsort();

                        for d1.each |&(k, v)| {
                            d1_flat.push((@copy *k, @copy *v));
                        }
                        d1_flat.qsort();

                        d0_flat < d1_flat
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
    fn le(&self, other: &Json) -> bool { !(*other).lt(&(*self)) }
    fn ge(&self, other: &Json) -> bool { !(*self).lt(other) }
    fn gt(&self, other: &Json) -> bool { (*other).lt(&(*self))  }
}

trait ToJson { fn to_json(&self) -> Json; }

impl ToJson for Json {
    fn to_json(&self) -> Json { copy *self }
}

impl ToJson for @Json {
    fn to_json(&self) -> Json { (**self).to_json() }
}

impl ToJson for int {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for i8 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for i16 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for i32 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for i64 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for uint {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for u8 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for u16 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for u32 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for u64 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for float {
    fn to_json(&self) -> Json { Number(*self) }
}

impl ToJson for f32 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for f64 {
    fn to_json(&self) -> Json { Number(*self as float) }
}

impl ToJson for () {
    fn to_json(&self) -> Json { Null }
}

impl ToJson for bool {
    fn to_json(&self) -> Json { Boolean(*self) }
}

impl ToJson for ~str {
    fn to_json(&self) -> Json { String(copy *self) }
}

impl ToJson for @~str {
    fn to_json(&self) -> Json { String(copy **self) }
}

impl<A:ToJson,B:ToJson> ToJson for (A, B) {
    fn to_json(&self) -> Json {
        match *self {
          (ref a, ref b) => {
            List(~[a.to_json(), b.to_json()])
          }
        }
    }
}

impl<A:ToJson,B:ToJson,C:ToJson> ToJson for (A, B, C) {
    fn to_json(&self) -> Json {
        match *self {
          (ref a, ref b, ref c) => {
            List(~[a.to_json(), b.to_json(), c.to_json()])
          }
        }
    }
}

impl<A:ToJson> ToJson for ~[A] {
    fn to_json(&self) -> Json { List(self.map(|elt| elt.to_json())) }
}

impl<A:ToJson + Copy> ToJson for HashMap<~str, A> {
    fn to_json(&self) -> Json {
        let mut d = HashMap::new();
        for self.each |&(key, value)| {
            d.insert(copy *key, value.to_json());
        }
        Object(~d)
    }
}

impl<A:ToJson> ToJson for Option<A> {
    fn to_json(&self) -> Json {
        match *self {
          None => Null,
          Some(ref value) => value.to_json()
        }
    }
}

impl to_str::ToStr for Json {
    fn to_str(&self) -> ~str { to_str(self) }
}

impl to_str::ToStr for Error {
    fn to_str(&self) -> ~str {
        fmt!("%u:%u: %s", self.line, self.col, *self.msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::prelude::*;
    use core::hashmap::HashMap;

    use std::serialize::Decodable;

    #[auto_encode]
    #[auto_decode]
    #[deriving(Eq)]
    enum Animal {
        Dog,
        Frog(~str, int)
    }

    #[auto_encode]
    #[auto_decode]
    #[deriving(Eq)]
    struct Inner {
        a: (),
        b: uint,
        c: ~[~str],
    }

    #[auto_encode]
    #[auto_decode]
    #[deriving(Eq)]
    struct Outer {
        inner: ~[Inner],
    }

    fn mk_object(items: &[(~str, Json)]) -> Json {
        let mut d = ~HashMap::new();

        for items.each |item| {
            match *item {
                (copy key, copy value) => { d.insert(key, value); },
            }
        };

        Object(d)
    }

    #[test]
    fn test_write_null() {
        assert_eq!(to_str(&Null), ~"null");
        assert_eq!(to_pretty_str(&Null), ~"null");
    }


    #[test]
    fn test_write_number() {
        assert_eq!(to_str(&Number(3f)), ~"3");
        assert_eq!(to_pretty_str(&Number(3f)), ~"3");

        assert_eq!(to_str(&Number(3.1f)), ~"3.1");
        assert_eq!(to_pretty_str(&Number(3.1f)), ~"3.1");

        assert_eq!(to_str(&Number(-1.5f)), ~"-1.5");
        assert_eq!(to_pretty_str(&Number(-1.5f)), ~"-1.5");

        assert_eq!(to_str(&Number(0.5f)), ~"0.5");
        assert_eq!(to_pretty_str(&Number(0.5f)), ~"0.5");
    }

    #[test]
    fn test_write_str() {
        assert_eq!(to_str(&String(~"")), ~"\"\"");
        assert_eq!(to_pretty_str(&String(~"")), ~"\"\"");

        assert_eq!(to_str(&String(~"foo")), ~"\"foo\"");
        assert_eq!(to_pretty_str(&String(~"foo")), ~"\"foo\"");
    }

    #[test]
    fn test_write_bool() {
        assert_eq!(to_str(&Boolean(true)), ~"true");
        assert_eq!(to_pretty_str(&Boolean(true)), ~"true");

        assert_eq!(to_str(&Boolean(false)), ~"false");
        assert_eq!(to_pretty_str(&Boolean(false)), ~"false");
    }

    #[test]
    fn test_write_list() {
        assert_eq!(to_str(&List(~[])), ~"[]");
        assert_eq!(to_pretty_str(&List(~[])), ~"[]");

        assert_eq!(to_str(&List(~[Boolean(true)])), ~"[true]");
        assert_eq!(
            to_pretty_str(&List(~[Boolean(true)])),
            ~"\
            [\n  \
                true\n\
            ]"
        );

        assert_eq!(to_str(&List(~[
            Boolean(false),
            Null,
            List(~[String(~"foo\nbar"), Number(3.5f)])
        ])), ~"[false,null,[\"foo\\nbar\",3.5]]");
        assert_eq!(
            to_pretty_str(&List(~[
                Boolean(false),
                Null,
                List(~[String(~"foo\nbar"), Number(3.5f)])
            ])),
            ~"\
            [\n  \
                false,\n  \
                null,\n  \
                [\n    \
                    \"foo\\nbar\",\n    \
                    3.5\n  \
                ]\n\
            ]"
        );
    }

    #[test]
    fn test_write_object() {
        assert_eq!(to_str(&mk_object(~[])), ~"{}");
        assert_eq!(to_pretty_str(&mk_object(~[])), ~"{}");

        assert_eq!(
            to_str(&mk_object(~[(~"a", Boolean(true))])),
            ~"{\"a\":true}"
        );
        assert_eq!(
            to_pretty_str(&mk_object(~[(~"a", Boolean(true))])),
            ~"\
            {\n  \
                \"a\": true\n\
            }"
        );

        assert_eq!(
            to_str(&mk_object(~[
                (~"b", List(~[
                    mk_object(~[(~"c", String(~"\x0c\r"))]),
                    mk_object(~[(~"d", String(~""))])
                ]))
            ])),
            ~"{\
                \"b\":[\
                    {\"c\":\"\\f\\r\"},\
                    {\"d\":\"\"}\
                ]\
            }"
        );
        assert_eq!(
            to_pretty_str(&mk_object(~[
                (~"b", List(~[
                    mk_object(~[(~"c", String(~"\x0c\r"))]),
                    mk_object(~[(~"d", String(~""))])
                ]))
            ])),
            ~"\
            {\n  \
                \"b\": [\n    \
                    {\n      \
                        \"c\": \"\\f\\r\"\n    \
                    },\n    \
                    {\n      \
                        \"d\": \"\"\n    \
                    }\n  \
                ]\n\
            }"
        );

        let a = mk_object(~[
            (~"a", Boolean(true)),
            (~"b", List(~[
                mk_object(~[(~"c", String(~"\x0c\r"))]),
                mk_object(~[(~"d", String(~""))])
            ]))
        ]);

        // We can't compare the strings directly because the object fields be
        // printed in a different order.
        assert_eq!(copy a, from_str(to_str(&a)).unwrap());
        assert_eq!(copy a, from_str(to_pretty_str(&a)).unwrap());
    }

    #[test]
    fn test_write_enum() {
        let animal = Dog;
        assert_eq!(
            do io::with_str_writer |wr| {
                let encoder = Encoder(wr);
                animal.encode(&encoder);
            },
            ~"\"Dog\""
        );
        assert_eq!(
            do io::with_str_writer |wr| {
                let encoder = PrettyEncoder(wr);
                animal.encode(&encoder);
            },
            ~"\"Dog\""
        );

        let animal = Frog(~"Henry", 349);
        assert_eq!(
            do io::with_str_writer |wr| {
                let encoder = Encoder(wr);
                animal.encode(&encoder);
            },
            ~"[\"Frog\",\"Henry\",349]"
        );
        assert_eq!(
            do io::with_str_writer |wr| {
                let encoder = PrettyEncoder(wr);
                animal.encode(&encoder);
            },
            ~"\
            [\n  \
                \"Frog\",\n  \
                \"Henry\",\n  \
                349\n\
            ]"
        );
    }

    #[test]
    fn test_write_some() {
        let value = Some(~"jodhpurs");
        let s = do io::with_str_writer |wr| {
            let encoder = Encoder(wr);
            value.encode(&encoder);
        };
        assert_eq!(s, ~"\"jodhpurs\"");

        let value = Some(~"jodhpurs");
        let s = do io::with_str_writer |wr| {
            let encoder = PrettyEncoder(wr);
            value.encode(&encoder);
        };
        assert_eq!(s, ~"\"jodhpurs\"");
    }

    #[test]
    fn test_write_none() {
        let value: Option<~str> = None;
        let s = do io::with_str_writer |wr| {
            let encoder = Encoder(wr);
            value.encode(&encoder);
        };
        assert_eq!(s, ~"null");

        let s = do io::with_str_writer |wr| {
            let encoder = Encoder(wr);
            value.encode(&encoder);
        };
        assert_eq!(s, ~"null");
    }

    #[test]
    fn test_trailing_characters() {
        assert_eq!(from_str(~"nulla"),
            Err(Error {line: 1u, col: 5u, msg: @~"trailing characters"}));
        assert_eq!(from_str(~"truea"),
            Err(Error {line: 1u, col: 5u, msg: @~"trailing characters"}));
        assert_eq!(from_str(~"falsea"),
            Err(Error {line: 1u, col: 6u, msg: @~"trailing characters"}));
        assert_eq!(from_str(~"1a"),
            Err(Error {line: 1u, col: 2u, msg: @~"trailing characters"}));
        assert_eq!(from_str(~"[]a"),
            Err(Error {line: 1u, col: 3u, msg: @~"trailing characters"}));
        assert_eq!(from_str(~"{}a"),
            Err(Error {line: 1u, col: 3u, msg: @~"trailing characters"}));
    }

    #[test]
    fn test_read_identifiers() {
        assert_eq!(from_str(~"n"),
            Err(Error {line: 1u, col: 2u, msg: @~"invalid syntax"}));
        assert_eq!(from_str(~"nul"),
            Err(Error {line: 1u, col: 4u, msg: @~"invalid syntax"}));

        assert_eq!(from_str(~"t"),
            Err(Error {line: 1u, col: 2u, msg: @~"invalid syntax"}));
        assert_eq!(from_str(~"truz"),
            Err(Error {line: 1u, col: 4u, msg: @~"invalid syntax"}));

        assert_eq!(from_str(~"f"),
            Err(Error {line: 1u, col: 2u, msg: @~"invalid syntax"}));
        assert_eq!(from_str(~"faz"),
            Err(Error {line: 1u, col: 3u, msg: @~"invalid syntax"}));

        assert_eq!(from_str(~"null"), Ok(Null));
        assert_eq!(from_str(~"true"), Ok(Boolean(true)));
        assert_eq!(from_str(~"false"), Ok(Boolean(false)));
        assert_eq!(from_str(~" null "), Ok(Null));
        assert_eq!(from_str(~" true "), Ok(Boolean(true)));
        assert_eq!(from_str(~" false "), Ok(Boolean(false)));
    }

    #[test]
    fn test_decode_identifiers() {
        let v: () = Decodable::decode(&Decoder(from_str(~"null").unwrap()));
        assert_eq!(v, ());

        let v: bool = Decodable::decode(&Decoder(from_str(~"true").unwrap()));
        assert_eq!(v, true);

        let v: bool = Decodable::decode(&Decoder(from_str(~"false").unwrap()));
        assert_eq!(v, false);
    }

    #[test]
    fn test_read_number() {
        assert_eq!(from_str(~"+"),
            Err(Error {line: 1u, col: 1u, msg: @~"invalid syntax"}));
        assert_eq!(from_str(~"."),
            Err(Error {line: 1u, col: 1u, msg: @~"invalid syntax"}));

        assert_eq!(from_str(~"-"),
            Err(Error {line: 1u, col: 2u, msg: @~"invalid number"}));
        assert_eq!(from_str(~"00"),
            Err(Error {line: 1u, col: 2u, msg: @~"invalid number"}));
        assert_eq!(from_str(~"1."),
            Err(Error {line: 1u, col: 3u, msg: @~"invalid number"}));
        assert_eq!(from_str(~"1e"),
            Err(Error {line: 1u, col: 3u, msg: @~"invalid number"}));
        assert_eq!(from_str(~"1e+"),
            Err(Error {line: 1u, col: 4u, msg: @~"invalid number"}));

        assert_eq!(from_str(~"3"), Ok(Number(3f)));
        assert_eq!(from_str(~"3.1"), Ok(Number(3.1f)));
        assert_eq!(from_str(~"-1.2"), Ok(Number(-1.2f)));
        assert_eq!(from_str(~"0.4"), Ok(Number(0.4f)));
        assert_eq!(from_str(~"0.4e5"), Ok(Number(0.4e5f)));
        assert_eq!(from_str(~"0.4e+15"), Ok(Number(0.4e15f)));
        assert_eq!(from_str(~"0.4e-01"), Ok(Number(0.4e-01f)));
        assert_eq!(from_str(~" 3 "), Ok(Number(3f)));
    }

    #[test]
    fn test_decode_numbers() {
        let v: float = Decodable::decode(&Decoder(from_str(~"3").unwrap()));
        assert_eq!(v, 3f);

        let v: float = Decodable::decode(&Decoder(from_str(~"3.1").unwrap()));
        assert_eq!(v, 3.1f);

        let v: float = Decodable::decode(&Decoder(from_str(~"-1.2").unwrap()));
        assert_eq!(v, -1.2f);

        let v: float = Decodable::decode(&Decoder(from_str(~"0.4").unwrap()));
        assert_eq!(v, 0.4f);

        let v: float = Decodable::decode(&Decoder(from_str(~"0.4e5").unwrap()));
        assert_eq!(v, 0.4e5f);

        let v: float = Decodable::decode(&Decoder(from_str(~"0.4e15").unwrap()));
        assert_eq!(v, 0.4e15f);

        let v: float = Decodable::decode(&Decoder(from_str(~"0.4e-01").unwrap()));
        assert_eq!(v, 0.4e-01f);
    }

    #[test]
    fn test_read_str() {
        assert_eq!(from_str(~"\""),
            Err(Error {line: 1u, col: 2u, msg: @~"EOF while parsing string"
        }));
        assert_eq!(from_str(~"\"lol"),
            Err(Error {line: 1u, col: 5u, msg: @~"EOF while parsing string"
        }));

        assert_eq!(from_str(~"\"\""), Ok(String(~"")));
        assert_eq!(from_str(~"\"foo\""), Ok(String(~"foo")));
        assert_eq!(from_str(~"\"\\\"\""), Ok(String(~"\"")));
        assert_eq!(from_str(~"\"\\b\""), Ok(String(~"\x08")));
        assert_eq!(from_str(~"\"\\n\""), Ok(String(~"\n")));
        assert_eq!(from_str(~"\"\\r\""), Ok(String(~"\r")));
        assert_eq!(from_str(~"\"\\t\""), Ok(String(~"\t")));
        assert_eq!(from_str(~" \"foo\" "), Ok(String(~"foo")));
        assert_eq!(from_str(~"\"\\u12ab\""), Ok(String(~"\u12ab")));
        assert_eq!(from_str(~"\"\\uAB12\""), Ok(String(~"\uAB12")));
    }

    #[test]
    fn test_decode_str() {
        let v: ~str = Decodable::decode(&Decoder(from_str(~"\"\"").unwrap()));
        assert_eq!(v, ~"");

        let v: ~str = Decodable::decode(&Decoder(from_str(~"\"foo\"").unwrap()));
        assert_eq!(v, ~"foo");

        let v: ~str = Decodable::decode(&Decoder(from_str(~"\"\\\"\"").unwrap()));
        assert_eq!(v, ~"\"");

        let v: ~str = Decodable::decode(&Decoder(from_str(~"\"\\b\"").unwrap()));
        assert_eq!(v, ~"\x08");

        let v: ~str = Decodable::decode(&Decoder(from_str(~"\"\\n\"").unwrap()));
        assert_eq!(v, ~"\n");

        let v: ~str = Decodable::decode(&Decoder(from_str(~"\"\\r\"").unwrap()));
        assert_eq!(v, ~"\r");

        let v: ~str = Decodable::decode(&Decoder(from_str(~"\"\\t\"").unwrap()));
        assert_eq!(v, ~"\t");

        let v: ~str = Decodable::decode(&Decoder(from_str(~"\"\\u12ab\"").unwrap()));
        assert_eq!(v, ~"\u12ab");

        let v: ~str = Decodable::decode(&Decoder(from_str(~"\"\\uAB12\"").unwrap()));
        assert_eq!(v, ~"\uAB12");
    }

    #[test]
    fn test_read_list() {
        assert_eq!(from_str(~"["),
            Err(Error {line: 1u, col: 2u, msg: @~"EOF while parsing value"}));
        assert_eq!(from_str(~"[1"),
            Err(Error {line: 1u, col: 3u, msg: @~"EOF while parsing list"}));
        assert_eq!(from_str(~"[1,"),
            Err(Error {line: 1u, col: 4u, msg: @~"EOF while parsing value"}));
        assert_eq!(from_str(~"[1,]"),
            Err(Error {line: 1u, col: 4u, msg: @~"invalid syntax"}));
        assert_eq!(from_str(~"[6 7]"),
            Err(Error {line: 1u, col: 4u, msg: @~"expected `,` or `]`"}));

        assert_eq!(from_str(~"[]"), Ok(List(~[])));
        assert_eq!(from_str(~"[ ]"), Ok(List(~[])));
        assert_eq!(from_str(~"[true]"), Ok(List(~[Boolean(true)])));
        assert_eq!(from_str(~"[ false ]"), Ok(List(~[Boolean(false)])));
        assert_eq!(from_str(~"[null]"), Ok(List(~[Null])));
        assert_eq!(from_str(~"[3, 1]"),
                     Ok(List(~[Number(3f), Number(1f)])));
        assert_eq!(from_str(~"\n[3, 2]\n"),
                     Ok(List(~[Number(3f), Number(2f)])));
        assert_eq!(from_str(~"[2, [4, 1]]"),
               Ok(List(~[Number(2f), List(~[Number(4f), Number(1f)])])));
    }

    #[test]
    fn test_decode_list() {
        let v: ~[()] = Decodable::decode(&Decoder(from_str(~"[]").unwrap()));
        assert_eq!(v, ~[]);

        let v: ~[()] = Decodable::decode(&Decoder(from_str(~"[null]").unwrap()));
        assert_eq!(v, ~[()]);


        let v: ~[bool] = Decodable::decode(&Decoder(from_str(~"[true]").unwrap()));
        assert_eq!(v, ~[true]);

        let v: ~[bool] = Decodable::decode(&Decoder(from_str(~"[true]").unwrap()));
        assert_eq!(v, ~[true]);

        let v: ~[int] = Decodable::decode(&Decoder(from_str(~"[3, 1]").unwrap()));
        assert_eq!(v, ~[3, 1]);

        let v: ~[~[uint]] = Decodable::decode(&Decoder(from_str(~"[[3], [1, 2]]").unwrap()));
        assert_eq!(v, ~[~[3], ~[1, 2]]);
    }

    #[test]
    fn test_read_object() {
        assert_eq!(from_str(~"{"),
            Err(Error {
                line: 1u,
                col: 2u,
                msg: @~"EOF while parsing object"}));
        assert_eq!(from_str(~"{ "),
            Err(Error {
                line: 1u,
                col: 3u,
                msg: @~"EOF while parsing object"}));
        assert_eq!(from_str(~"{1"),
            Err(Error {
                line: 1u,
                col: 2u,
                msg: @~"key must be a string"}));
        assert_eq!(from_str(~"{ \"a\""),
            Err(Error {
                line: 1u,
                col: 6u,
                msg: @~"EOF while parsing object"}));
        assert_eq!(from_str(~"{\"a\""),
            Err(Error {
                line: 1u,
                col: 5u,
                msg: @~"EOF while parsing object"}));
        assert_eq!(from_str(~"{\"a\" "),
            Err(Error {
                line: 1u,
                col: 6u,
                msg: @~"EOF while parsing object"}));

        assert_eq!(from_str(~"{\"a\" 1"),
            Err(Error {line: 1u, col: 6u, msg: @~"expected `:`"}));
        assert_eq!(from_str(~"{\"a\":"),
            Err(Error {line: 1u, col: 6u, msg: @~"EOF while parsing value"}));
        assert_eq!(from_str(~"{\"a\":1"),
            Err(Error {
                line: 1u,
                col: 7u,
                msg: @~"EOF while parsing object"}));
        assert_eq!(from_str(~"{\"a\":1 1"),
            Err(Error {line: 1u, col: 8u, msg: @~"expected `,` or `}`"}));
        assert_eq!(from_str(~"{\"a\":1,"),
            Err(Error {
                line: 1u,
                col: 8u,
                msg: @~"EOF while parsing object"}));

        assert_eq!(result::unwrap(from_str(~"{}")), mk_object(~[]));
        assert_eq!(result::unwrap(from_str(~"{\"a\": 3}")),
                  mk_object(~[(~"a", Number(3.0f))]));

        assert_eq!(result::unwrap(from_str(
                ~"{ \"a\": null, \"b\" : true }")),
                  mk_object(~[
                      (~"a", Null),
                      (~"b", Boolean(true))]));
        assert_eq!(result::unwrap(
                      from_str(~"\n{ \"a\": null, \"b\" : true }\n")),
                  mk_object(~[
                      (~"a", Null),
                      (~"b", Boolean(true))]));
        assert_eq!(result::unwrap(from_str(
                ~"{\"a\" : 1.0 ,\"b\": [ true ]}")),
                  mk_object(~[
                      (~"a", Number(1.0)),
                      (~"b", List(~[Boolean(true)]))
                  ]));
        assert_eq!(result::unwrap(from_str(
                      ~"{" +
                          ~"\"a\": 1.0, " +
                          ~"\"b\": [" +
                              ~"true," +
                              ~"\"foo\\nbar\", " +
                              ~"{ \"c\": {\"d\": null} } " +
                          ~"]" +
                      ~"}")),
                  mk_object(~[
                      (~"a", Number(1.0f)),
                      (~"b", List(~[
                          Boolean(true),
                          String(~"foo\nbar"),
                          mk_object(~[
                              (~"c", mk_object(~[(~"d", Null)]))
                          ])
                      ]))
                  ]));
    }

    #[test]
    fn test_decode_struct() {
        let s = ~"{
            \"inner\": [
                { \"a\": null, \"b\": 2, \"c\": [\"abc\", \"xyz\"] }
            ]
        }";
        let v: Outer = Decodable::decode(&Decoder(from_str(s).unwrap()));
        assert_eq!(
            v,
            Outer {
                inner: ~[
                    Inner { a: (), b: 2, c: ~[~"abc", ~"xyz"] }
                ]
            }
        );
    }

    #[test]
    fn test_decode_option() {
        let decoder = Decoder(from_str(~"null").unwrap());
        let value: Option<~str> = Decodable::decode(&decoder);
        assert_eq!(value, None);

        let decoder = Decoder(from_str(~"\"jodhpurs\"").unwrap());
        let value: Option<~str> = Decodable::decode(&decoder);
        assert_eq!(value, Some(~"jodhpurs"));
    }

    #[test]
    fn test_decode_enum() {
        let decoder = Decoder(from_str(~"\"Dog\"").unwrap());
        let value: Animal = Decodable::decode(&decoder);
        assert_eq!(value, Dog);

        let decoder = Decoder(from_str(~"[\"Frog\",\"Henry\",349]").unwrap());
        let value: Animal = Decodable::decode(&decoder);
        assert_eq!(value, Frog(~"Henry", 349));
    }

    #[test]
    fn test_decode_map() {
        let s = ~"{\"a\": \"Dog\", \"b\": [\"Frog\", \"Henry\", 349]}";
        let decoder = Decoder(from_str(s).unwrap());
        let mut map: HashMap<~str, Animal> = Decodable::decode(&decoder);

        assert_eq!(map.pop(&~"a"), Some(Dog));
        assert_eq!(map.pop(&~"b"), Some(Frog(~"Henry", 349)));
    }

    #[test]
    fn test_multiline_errors() {
        assert_eq!(from_str(~"{\n  \"foo\":\n \"bar\""),
            Err(Error {
                line: 3u,
                col: 8u,
                msg: @~"EOF while parsing object"}));
    }
}
