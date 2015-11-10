// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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

#![forbid(non_camel_case_types)]
#![allow(missing_docs)]

//! JSON parsing and serialization
//!
//! # What is JSON?
//!
//! JSON (JavaScript Object Notation) is a way to write data in Javascript.
//! Like XML, it allows to encode structured data in a text format that can be easily read by humans
//! Its simple syntax and native compatibility with JavaScript have made it a widely used format.
//!
//! Data types that can be encoded are JavaScript types (see the `Json` enum for more details):
//!
//! * `Boolean`: equivalent to rust's `bool`
//! * `Number`: equivalent to rust's `f64`
//! * `String`: equivalent to rust's `String`
//! * `Array`: equivalent to rust's `Vec<T>`, but also allowing objects of different types in the
//!   same array
//! * `Object`: equivalent to rust's `BTreeMap<String, json::Json>`
//! * `Null`
//!
//! An object is a series of string keys mapping to values, in `"key": value` format.
//! Arrays are enclosed in square brackets ([ ... ]) and objects in curly brackets ({ ... }).
//! A simple JSON document encoding a person, their age, address and phone numbers could look like
//!
//! ```ignore
//! {
//!     "FirstName": "John",
//!     "LastName": "Doe",
//!     "Age": 43,
//!     "Address": {
//!         "Street": "Downing Street 10",
//!         "City": "London",
//!         "Country": "Great Britain"
//!     },
//!     "PhoneNumbers": [
//!         "+44 1234567",
//!         "+44 2345678"
//!     ]
//! }
//! ```
//!
//! # Rust Type-based Encoding and Decoding
//!
//! Rust provides a mechanism for low boilerplate encoding & decoding of values to and from JSON via
//! the serialization API.
//! To be able to encode a piece of data, it must implement the `serialize::RustcEncodable` trait.
//! To be able to decode a piece of data, it must implement the `serialize::RustcDecodable` trait.
//! The Rust compiler provides an annotation to automatically generate the code for these traits:
//! `#[derive(RustcDecodable, RustcEncodable)]`
//!
//! The JSON API provides an enum `json::Json` and a trait `ToJson` to encode objects.
//! The `ToJson` trait provides a `to_json` method to convert an object into a `json::Json` value.
//! A `json::Json` value can be encoded as a string or buffer using the functions described above.
//! You can also use the `json::Encoder` object, which implements the `Encoder` trait.
//!
//! When using `ToJson` the `RustcEncodable` trait implementation is not mandatory.
//!
//! # Examples of use
//!
//! ## Using Autoserialization
//!
//! Create a struct called `TestStruct` and serialize and deserialize it to and from JSON using the
//! serialization API, using the derived serialization code.
//!
//! ```rust
//! # #![feature(rustc_private)]
//! extern crate serialize as rustc_serialize; // for the deriving below
//! use rustc_serialize::json;
//!
//! // Automatically generate `Decodable` and `Encodable` trait implementations
//! #[derive(RustcDecodable, RustcEncodable)]
//! pub struct TestStruct  {
//!     data_int: u8,
//!     data_str: String,
//!     data_vector: Vec<u8>,
//! }
//!
//! fn main() {
//!     let object = TestStruct {
//!         data_int: 1,
//!         data_str: "homura".to_string(),
//!         data_vector: vec![2,3,4,5],
//!     };
//!
//!     // Serialize using `json::encode`
//!     let encoded = json::encode(&object).unwrap();
//!
//!     // Deserialize using `json::decode`
//!     let decoded: TestStruct = json::decode(&encoded[..]).unwrap();
//! }
//! ```
//!
//! ## Using the `ToJson` trait
//!
//! The examples above use the `ToJson` trait to generate the JSON string, which is required
//! for custom mappings.
//!
//! ### Simple example of `ToJson` usage
//!
//! ```rust
//! # #![feature(rustc_private)]
//! extern crate serialize;
//! use serialize::json::{self, ToJson, Json};
//!
//! // A custom data structure
//! struct ComplexNum {
//!     a: f64,
//!     b: f64,
//! }
//!
//! // JSON value representation
//! impl ToJson for ComplexNum {
//!     fn to_json(&self) -> Json {
//!         Json::String(format!("{}+{}i", self.a, self.b))
//!     }
//! }
//!
//! // Only generate `RustcEncodable` trait implementation
//! #[derive(Encodable)]
//! pub struct ComplexNumRecord {
//!     uid: u8,
//!     dsc: String,
//!     val: Json,
//! }
//!
//! fn main() {
//!     let num = ComplexNum { a: 0.0001, b: 12.539 };
//!     let data: String = json::encode(&ComplexNumRecord{
//!         uid: 1,
//!         dsc: "test".to_string(),
//!         val: num.to_json(),
//!     }).unwrap();
//!     println!("data: {}", data);
//!     // data: {"uid":1,"dsc":"test","val":"0.0001+12.539i"};
//! }
//! ```
//!
//! ### Verbose example of `ToJson` usage
//!
//! ```rust
//! # #![feature(rustc_private)]
//! extern crate serialize;
//! use std::collections::BTreeMap;
//! use serialize::json::{self, Json, ToJson};
//!
//! // Only generate `Decodable` trait implementation
//! #[derive(Decodable)]
//! pub struct TestStruct {
//!     data_int: u8,
//!     data_str: String,
//!     data_vector: Vec<u8>,
//! }
//!
//! // Specify encoding method manually
//! impl ToJson for TestStruct {
//!     fn to_json(&self) -> Json {
//!         let mut d = BTreeMap::new();
//!         // All standard types implement `to_json()`, so use it
//!         d.insert("data_int".to_string(), self.data_int.to_json());
//!         d.insert("data_str".to_string(), self.data_str.to_json());
//!         d.insert("data_vector".to_string(), self.data_vector.to_json());
//!         Json::Object(d)
//!     }
//! }
//!
//! fn main() {
//!     // Serialize using `ToJson`
//!     let input_data = TestStruct {
//!         data_int: 1,
//!         data_str: "madoka".to_string(),
//!         data_vector: vec![2,3,4,5],
//!     };
//!     let json_obj: Json = input_data.to_json();
//!     let json_str: String = json_obj.to_string();
//!
//!     // Deserialize like before
//!     let decoded: TestStruct = json::decode(&json_str).unwrap();
//! }
//! ```

use self::JsonEvent::*;
use self::ErrorCode::*;
use self::ParserError::*;
use self::DecoderError::*;
use self::ParserState::*;
use self::InternalStackElement::*;

use std::collections::{HashMap, BTreeMap};
use std::io::prelude::*;
use std::io;
use std::mem::swap;
use std::num::FpCategory as Fp;
use std::ops::Index;
use std::str::FromStr;
use std::string;
use std::{char, f64, fmt, str};
use std;

use Encodable;

/// Represents a json value
#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub enum Json {
    I64(i64),
    U64(u64),
    F64(f64),
    String(string::String),
    Boolean(bool),
    Array(self::Array),
    Object(self::Object),
    Null,
}

pub type Array = Vec<Json>;
pub type Object = BTreeMap<string::String, Json>;

pub struct PrettyJson<'a> { inner: &'a Json }

pub struct AsJson<'a, T: 'a> { inner: &'a T }
pub struct AsPrettyJson<'a, T: 'a> { inner: &'a T, indent: Option<usize> }

/// The errors that can arise while parsing a JSON stream.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ErrorCode {
    InvalidSyntax,
    InvalidNumber,
    EOFWhileParsingObject,
    EOFWhileParsingArray,
    EOFWhileParsingValue,
    EOFWhileParsingString,
    KeyMustBeAString,
    ExpectedColon,
    TrailingCharacters,
    TrailingComma,
    InvalidEscape,
    InvalidUnicodeCodePoint,
    LoneLeadingSurrogateInHexEscape,
    UnexpectedEndOfHexEscape,
    UnrecognizedHex,
    NotFourDigit,
    NotUtf8,
}

#[derive(Clone, PartialEq, Debug)]
pub enum ParserError {
    /// msg, line, col
    SyntaxError(ErrorCode, usize, usize),
    IoError(io::ErrorKind, String),
}

// Builder and Parser have the same errors.
pub type BuilderError = ParserError;

#[derive(Clone, PartialEq, Debug)]
pub enum DecoderError {
    ParseError(ParserError),
    ExpectedError(string::String, string::String),
    MissingFieldError(string::String),
    UnknownVariantError(string::String),
    ApplicationError(string::String)
}

#[derive(Copy, Clone, Debug)]
pub enum EncoderError {
    FmtError(fmt::Error),
    BadHashmapKey,
}

/// Returns a readable error string for a given error code.
pub fn error_str(error: ErrorCode) -> &'static str {
    match error {
        InvalidSyntax => "invalid syntax",
        InvalidNumber => "invalid number",
        EOFWhileParsingObject => "EOF While parsing object",
        EOFWhileParsingArray => "EOF While parsing array",
        EOFWhileParsingValue => "EOF While parsing value",
        EOFWhileParsingString => "EOF While parsing string",
        KeyMustBeAString => "key must be a string",
        ExpectedColon => "expected `:`",
        TrailingCharacters => "trailing characters",
        TrailingComma => "trailing comma",
        InvalidEscape => "invalid escape",
        UnrecognizedHex => "invalid \\u{ esc}ape (unrecognized hex)",
        NotFourDigit => "invalid \\u{ esc}ape (not four digits)",
        NotUtf8 => "contents not utf-8",
        InvalidUnicodeCodePoint => "invalid Unicode code point",
        LoneLeadingSurrogateInHexEscape => "lone leading surrogate in hex escape",
        UnexpectedEndOfHexEscape => "unexpected end of hex escape",
    }
}

/// Shortcut function to decode a JSON `&str` into an object
pub fn decode<T: ::Decodable>(s: &str) -> DecodeResult<T> {
    let json = match from_str(s) {
        Ok(x) => x,
        Err(e) => return Err(ParseError(e))
    };

    let mut decoder = Decoder::new(json);
    ::Decodable::decode(&mut decoder)
}

/// Shortcut function to encode a `T` into a JSON `String`
pub fn encode<T: ::Encodable>(object: &T) -> Result<string::String, EncoderError> {
    let mut s = String::new();
    {
        let mut encoder = Encoder::new(&mut s);
        try!(object.encode(&mut encoder));
    }
    Ok(s)
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        error_str(*self).fmt(f)
    }
}

fn io_error_to_error(io: io::Error) -> ParserError {
    IoError(io.kind(), io.to_string())
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // FIXME this should be a nicer error
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // FIXME this should be a nicer error
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for DecoderError {
    fn description(&self) -> &str { "decoder error" }
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // FIXME this should be a nicer error
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for EncoderError {
    fn description(&self) -> &str { "encoder error" }
}

impl From<fmt::Error> for EncoderError {
    fn from(err: fmt::Error) -> EncoderError { EncoderError::FmtError(err) }
}

pub type EncodeResult = Result<(), EncoderError>;
pub type DecodeResult<T> = Result<T, DecoderError>;

fn escape_str(wr: &mut fmt::Write, v: &str) -> EncodeResult {
    try!(wr.write_str("\""));

    let mut start = 0;

    for (i, byte) in v.bytes().enumerate() {
        let escaped = match byte {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            b'\x00' => "\\u0000",
            b'\x01' => "\\u0001",
            b'\x02' => "\\u0002",
            b'\x03' => "\\u0003",
            b'\x04' => "\\u0004",
            b'\x05' => "\\u0005",
            b'\x06' => "\\u0006",
            b'\x07' => "\\u0007",
            b'\x08' => "\\b",
            b'\t' => "\\t",
            b'\n' => "\\n",
            b'\x0b' => "\\u000b",
            b'\x0c' => "\\f",
            b'\r' => "\\r",
            b'\x0e' => "\\u000e",
            b'\x0f' => "\\u000f",
            b'\x10' => "\\u0010",
            b'\x11' => "\\u0011",
            b'\x12' => "\\u0012",
            b'\x13' => "\\u0013",
            b'\x14' => "\\u0014",
            b'\x15' => "\\u0015",
            b'\x16' => "\\u0016",
            b'\x17' => "\\u0017",
            b'\x18' => "\\u0018",
            b'\x19' => "\\u0019",
            b'\x1a' => "\\u001a",
            b'\x1b' => "\\u001b",
            b'\x1c' => "\\u001c",
            b'\x1d' => "\\u001d",
            b'\x1e' => "\\u001e",
            b'\x1f' => "\\u001f",
            b'\x7f' => "\\u007f",
            _ => { continue; }
        };

        if start < i {
            try!(wr.write_str(&v[start..i]));
        }

        try!(wr.write_str(escaped));

        start = i + 1;
    }

    if start != v.len() {
        try!(wr.write_str(&v[start..]));
    }

    try!(wr.write_str("\""));
    Ok(())
}

fn escape_char(writer: &mut fmt::Write, v: char) -> EncodeResult {
    let mut buf = [0; 4];
    let n = v.encode_utf8(&mut buf).unwrap();
    let buf = unsafe { str::from_utf8_unchecked(&buf[..n]) };
    escape_str(writer, buf)
}

fn spaces(wr: &mut fmt::Write, mut n: usize) -> EncodeResult {
    const BUF: &'static str = "                ";

    while n >= BUF.len() {
        try!(wr.write_str(BUF));
        n -= BUF.len();
    }

    if n > 0 {
        try!(wr.write_str(&BUF[..n]));
    }
    Ok(())
}

fn fmt_number_or_null(v: f64) -> string::String {
    match v.classify() {
        Fp::Nan | Fp::Infinite => string::String::from("null"),
        _ if v.fract() != 0f64 => v.to_string(),
        _ => v.to_string() + ".0",
    }
}

/// A structure for implementing serialization to JSON.
pub struct Encoder<'a> {
    writer: &'a mut (fmt::Write+'a),
    is_emitting_map_key: bool,
}

impl<'a> Encoder<'a> {
    /// Creates a new JSON encoder whose output will be written to the writer
    /// specified.
    pub fn new(writer: &'a mut fmt::Write) -> Encoder<'a> {
        Encoder { writer: writer, is_emitting_map_key: false, }
    }
}

macro_rules! emit_enquoted_if_mapkey {
    ($enc:ident,$e:expr) => {
        if $enc.is_emitting_map_key {
            try!(write!($enc.writer, "\"{}\"", $e));
            Ok(())
        } else {
            try!(write!($enc.writer, "{}", $e));
            Ok(())
        }
    }
}

impl<'a> ::Encoder for Encoder<'a> {
    type Error = EncoderError;

    fn emit_nil(&mut self) -> EncodeResult {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        try!(write!(self.writer, "null"));
        Ok(())
    }

    fn emit_uint(&mut self, v: usize) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_u64(&mut self, v: u64) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_u32(&mut self, v: u32) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_u16(&mut self, v: u16) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_u8(&mut self, v: u8) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }

    fn emit_int(&mut self, v: isize) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_i64(&mut self, v: i64) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_i32(&mut self, v: i32) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_i16(&mut self, v: i16) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_i8(&mut self, v: i8) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }

    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if v {
            try!(write!(self.writer, "true"));
        } else {
            try!(write!(self.writer, "false"));
        }
        Ok(())
    }

    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, fmt_number_or_null(v))
    }
    fn emit_f32(&mut self, v: f32) -> EncodeResult {
        self.emit_f64(v as f64)
    }

    fn emit_char(&mut self, v: char) -> EncodeResult {
        escape_char(self.writer, v)
    }
    fn emit_str(&mut self, v: &str) -> EncodeResult {
        escape_str(self.writer, v)
    }

    fn emit_enum<F>(&mut self, _name: &str, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        f(self)
    }

    fn emit_enum_variant<F>(&mut self,
                            name: &str,
                            _id: usize,
                            cnt: usize,
                            f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        // enums are encoded as strings or objects
        // Bunny => "Bunny"
        // Kangaroo(34,"William") => {"variant": "Kangaroo", "fields": [34,"William"]}
        if cnt == 0 {
            escape_str(self.writer, name)
        } else {
            if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
            try!(write!(self.writer, "{{\"variant\":"));
            try!(escape_str(self.writer, name));
            try!(write!(self.writer, ",\"fields\":["));
            try!(f(self));
            try!(write!(self.writer, "]}}"));
            Ok(())
        }
    }

    fn emit_enum_variant_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if idx != 0 {
            try!(write!(self.writer, ","));
        }
        f(self)
    }

    fn emit_enum_struct_variant<F>(&mut self,
                                   name: &str,
                                   id: usize,
                                   cnt: usize,
                                   f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_enum_variant(name, id, cnt, f)
    }

    fn emit_enum_struct_variant_field<F>(&mut self,
                                         _: &str,
                                         idx: usize,
                                         f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_enum_variant_arg(idx, f)
    }

    fn emit_struct<F>(&mut self, _: &str, _: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        try!(write!(self.writer, "{{"));
        try!(f(self));
        try!(write!(self.writer, "}}"));
        Ok(())
    }

    fn emit_struct_field<F>(&mut self, name: &str, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if idx != 0 { try!(write!(self.writer, ",")); }
        try!(escape_str(self.writer, name));
        try!(write!(self.writer, ":"));
        f(self)
    }

    fn emit_tuple<F>(&mut self, len: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_seq_elt(idx, f)
    }

    fn emit_tuple_struct<F>(&mut self, _name: &str, len: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_seq(len, f)
    }
    fn emit_tuple_struct_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_seq_elt(idx, f)
    }

    fn emit_option<F>(&mut self, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        f(self)
    }
    fn emit_option_none(&mut self) -> EncodeResult {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_nil()
    }
    fn emit_option_some<F>(&mut self, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        f(self)
    }

    fn emit_seq<F>(&mut self, _len: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        try!(write!(self.writer, "["));
        try!(f(self));
        try!(write!(self.writer, "]"));
        Ok(())
    }

    fn emit_seq_elt<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if idx != 0 {
            try!(write!(self.writer, ","));
        }
        f(self)
    }

    fn emit_map<F>(&mut self, _len: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        try!(write!(self.writer, "{{"));
        try!(f(self));
        try!(write!(self.writer, "}}"));
        Ok(())
    }

    fn emit_map_elt_key<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if idx != 0 { try!(write!(self.writer, ",")) }
        self.is_emitting_map_key = true;
        try!(f(self));
        self.is_emitting_map_key = false;
        Ok(())
    }

    fn emit_map_elt_val<F>(&mut self, _idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        try!(write!(self.writer, ":"));
        f(self)
    }
}

/// Another encoder for JSON, but prints out human-readable JSON instead of
/// compact data
pub struct PrettyEncoder<'a> {
    writer: &'a mut (fmt::Write+'a),
    curr_indent: usize,
    indent: usize,
    is_emitting_map_key: bool,
}

impl<'a> PrettyEncoder<'a> {
    /// Creates a new encoder whose output will be written to the specified writer
    pub fn new(writer: &'a mut fmt::Write) -> PrettyEncoder<'a> {
        PrettyEncoder {
            writer: writer,
            curr_indent: 0,
            indent: 2,
            is_emitting_map_key: false,
        }
    }

    /// Set the number of spaces to indent for each level.
    /// This is safe to set during encoding.
    pub fn set_indent(&mut self, indent: usize) {
        // self.indent very well could be 0 so we need to use checked division.
        let level = self.curr_indent.checked_div(self.indent).unwrap_or(0);
        self.indent = indent;
        self.curr_indent = level * self.indent;
    }
}

impl<'a> ::Encoder for PrettyEncoder<'a> {
    type Error = EncoderError;

    fn emit_nil(&mut self) -> EncodeResult {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        try!(write!(self.writer, "null"));
        Ok(())
    }

    fn emit_uint(&mut self, v: usize) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_u64(&mut self, v: u64) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_u32(&mut self, v: u32) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_u16(&mut self, v: u16) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_u8(&mut self, v: u8) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }

    fn emit_int(&mut self, v: isize) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_i64(&mut self, v: i64) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_i32(&mut self, v: i32) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_i16(&mut self, v: i16) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }
    fn emit_i8(&mut self, v: i8) -> EncodeResult { emit_enquoted_if_mapkey!(self, v) }

    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if v {
            try!(write!(self.writer, "true"));
        } else {
            try!(write!(self.writer, "false"));
        }
        Ok(())
    }

    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, fmt_number_or_null(v))
    }
    fn emit_f32(&mut self, v: f32) -> EncodeResult {
        self.emit_f64(v as f64)
    }

    fn emit_char(&mut self, v: char) -> EncodeResult {
        escape_char(self.writer, v)
    }
    fn emit_str(&mut self, v: &str) -> EncodeResult {
        escape_str(self.writer, v)
    }

    fn emit_enum<F>(&mut self, _name: &str, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        f(self)
    }

    fn emit_enum_variant<F>(&mut self,
                            name: &str,
                            _id: usize,
                            cnt: usize,
                            f: F)
                            -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if cnt == 0 {
            escape_str(self.writer, name)
        } else {
            if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
            try!(write!(self.writer, "{{\n"));
            self.curr_indent += self.indent;
            try!(spaces(self.writer, self.curr_indent));
            try!(write!(self.writer, "\"variant\": "));
            try!(escape_str(self.writer, name));
            try!(write!(self.writer, ",\n"));
            try!(spaces(self.writer, self.curr_indent));
            try!(write!(self.writer, "\"fields\": [\n"));
            self.curr_indent += self.indent;
            try!(f(self));
            self.curr_indent -= self.indent;
            try!(write!(self.writer, "\n"));
            try!(spaces(self.writer, self.curr_indent));
            self.curr_indent -= self.indent;
            try!(write!(self.writer, "]\n"));
            try!(spaces(self.writer, self.curr_indent));
            try!(write!(self.writer, "}}"));
            Ok(())
        }
    }

    fn emit_enum_variant_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if idx != 0 {
            try!(write!(self.writer, ",\n"));
        }
        try!(spaces(self.writer, self.curr_indent));
        f(self)
    }

    fn emit_enum_struct_variant<F>(&mut self,
                                   name: &str,
                                   id: usize,
                                   cnt: usize,
                                   f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_enum_variant(name, id, cnt, f)
    }

    fn emit_enum_struct_variant_field<F>(&mut self,
                                         _: &str,
                                         idx: usize,
                                         f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_enum_variant_arg(idx, f)
    }


    fn emit_struct<F>(&mut self, _: &str, len: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if len == 0 {
            try!(write!(self.writer, "{{}}"));
        } else {
            try!(write!(self.writer, "{{"));
            self.curr_indent += self.indent;
            try!(f(self));
            self.curr_indent -= self.indent;
            try!(write!(self.writer, "\n"));
            try!(spaces(self.writer, self.curr_indent));
            try!(write!(self.writer, "}}"));
        }
        Ok(())
    }

    fn emit_struct_field<F>(&mut self, name: &str, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if idx == 0 {
            try!(write!(self.writer, "\n"));
        } else {
            try!(write!(self.writer, ",\n"));
        }
        try!(spaces(self.writer, self.curr_indent));
        try!(escape_str(self.writer, name));
        try!(write!(self.writer, ": "));
        f(self)
    }

    fn emit_tuple<F>(&mut self, len: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_seq_elt(idx, f)
    }

    fn emit_tuple_struct<F>(&mut self, _: &str, len: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_seq(len, f)
    }
    fn emit_tuple_struct_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_seq_elt(idx, f)
    }

    fn emit_option<F>(&mut self, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        f(self)
    }
    fn emit_option_none(&mut self) -> EncodeResult {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        self.emit_nil()
    }
    fn emit_option_some<F>(&mut self, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        f(self)
    }

    fn emit_seq<F>(&mut self, len: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if len == 0 {
            try!(write!(self.writer, "[]"));
        } else {
            try!(write!(self.writer, "["));
            self.curr_indent += self.indent;
            try!(f(self));
            self.curr_indent -= self.indent;
            try!(write!(self.writer, "\n"));
            try!(spaces(self.writer, self.curr_indent));
            try!(write!(self.writer, "]"));
        }
        Ok(())
    }

    fn emit_seq_elt<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if idx == 0 {
            try!(write!(self.writer, "\n"));
        } else {
            try!(write!(self.writer, ",\n"));
        }
        try!(spaces(self.writer, self.curr_indent));
        f(self)
    }

    fn emit_map<F>(&mut self, len: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if len == 0 {
            try!(write!(self.writer, "{{}}"));
        } else {
            try!(write!(self.writer, "{{"));
            self.curr_indent += self.indent;
            try!(f(self));
            self.curr_indent -= self.indent;
            try!(write!(self.writer, "\n"));
            try!(spaces(self.writer, self.curr_indent));
            try!(write!(self.writer, "}}"));
        }
        Ok(())
    }

    fn emit_map_elt_key<F>(&mut self, idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        if idx == 0 {
            try!(write!(self.writer, "\n"));
        } else {
            try!(write!(self.writer, ",\n"));
        }
        try!(spaces(self.writer, self.curr_indent));
        self.is_emitting_map_key = true;
        try!(f(self));
        self.is_emitting_map_key = false;
        Ok(())
    }

    fn emit_map_elt_val<F>(&mut self, _idx: usize, f: F) -> EncodeResult where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key { return Err(EncoderError::BadHashmapKey); }
        try!(write!(self.writer, ": "));
        f(self)
    }
}

impl Encodable for Json {
    fn encode<E: ::Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
        match *self {
            Json::I64(v) => v.encode(e),
            Json::U64(v) => v.encode(e),
            Json::F64(v) => v.encode(e),
            Json::String(ref v) => v.encode(e),
            Json::Boolean(v) => v.encode(e),
            Json::Array(ref v) => v.encode(e),
            Json::Object(ref v) => v.encode(e),
            Json::Null => e.emit_nil(),
        }
    }
}

/// Create an `AsJson` wrapper which can be used to print a value as JSON
/// on-the-fly via `write!`
pub fn as_json<T>(t: &T) -> AsJson<T> {
    AsJson { inner: t }
}

/// Create an `AsPrettyJson` wrapper which can be used to print a value as JSON
/// on-the-fly via `write!`
pub fn as_pretty_json<T>(t: &T) -> AsPrettyJson<T> {
    AsPrettyJson { inner: t, indent: None }
}

impl Json {
    /// Borrow this json object as a pretty object to generate a pretty
    /// representation for it via `Display`.
    pub fn pretty(&self) -> PrettyJson {
        PrettyJson { inner: self }
    }

     /// If the Json value is an Object, returns the value associated with the provided key.
    /// Otherwise, returns None.
    pub fn find<'a>(&'a self, key: &str) -> Option<&'a Json>{
        match *self {
            Json::Object(ref map) => map.get(key),
            _ => None
        }
    }

    /// Attempts to get a nested Json Object for each key in `keys`.
    /// If any key is found not to exist, find_path will return None.
    /// Otherwise, it will return the Json value associated with the final key.
    pub fn find_path<'a>(&'a self, keys: &[&str]) -> Option<&'a Json>{
        let mut target = self;
        for key in keys {
            match target.find(*key) {
                Some(t) => { target = t; },
                None => return None
            }
        }
        Some(target)
    }

    /// If the Json value is an Object, performs a depth-first search until
    /// a value associated with the provided key is found. If no value is found
    /// or the Json value is not an Object, returns None.
    pub fn search<'a>(&'a self, key: &str) -> Option<&'a Json> {
        match self {
            &Json::Object(ref map) => {
                match map.get(key) {
                    Some(json_value) => Some(json_value),
                    None => {
                        for (_, v) in map {
                            match v.search(key) {
                                x if x.is_some() => return x,
                                _ => ()
                            }
                        }
                        None
                    }
                }
            },
            _ => None
        }
    }

    /// Returns true if the Json value is an Object. Returns false otherwise.
    pub fn is_object(&self) -> bool {
        self.as_object().is_some()
    }

    /// If the Json value is an Object, returns the associated BTreeMap.
    /// Returns None otherwise.
    pub fn as_object(&self) -> Option<&Object> {
        match *self {
            Json::Object(ref map) => Some(map),
            _ => None
        }
    }

    /// Returns true if the Json value is an Array. Returns false otherwise.
    pub fn is_array(&self) -> bool {
        self.as_array().is_some()
    }

    /// If the Json value is an Array, returns the associated vector.
    /// Returns None otherwise.
    pub fn as_array(&self) -> Option<&Array> {
        match *self {
            Json::Array(ref array) => Some(&*array),
            _ => None
        }
    }

    /// Returns true if the Json value is a String. Returns false otherwise.
    pub fn is_string(&self) -> bool {
        self.as_string().is_some()
    }

    /// If the Json value is a String, returns the associated str.
    /// Returns None otherwise.
    pub fn as_string(&self) -> Option<&str> {
        match *self {
            Json::String(ref s) => Some(&s[..]),
            _ => None
        }
    }

    /// Returns true if the Json value is a Number. Returns false otherwise.
    pub fn is_number(&self) -> bool {
        match *self {
            Json::I64(_) | Json::U64(_) | Json::F64(_) => true,
            _ => false,
        }
    }

    /// Returns true if the Json value is a i64. Returns false otherwise.
    pub fn is_i64(&self) -> bool {
        match *self {
            Json::I64(_) => true,
            _ => false,
        }
    }

    /// Returns true if the Json value is a u64. Returns false otherwise.
    pub fn is_u64(&self) -> bool {
        match *self {
            Json::U64(_) => true,
            _ => false,
        }
    }

    /// Returns true if the Json value is a f64. Returns false otherwise.
    pub fn is_f64(&self) -> bool {
        match *self {
            Json::F64(_) => true,
            _ => false,
        }
    }

    /// If the Json value is a number, return or cast it to a i64.
    /// Returns None otherwise.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Json::I64(n) => Some(n),
            Json::U64(n) => Some(n as i64),
            _ => None
        }
    }

    /// If the Json value is a number, return or cast it to a u64.
    /// Returns None otherwise.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Json::I64(n) => Some(n as u64),
            Json::U64(n) => Some(n),
            _ => None
        }
    }

    /// If the Json value is a number, return or cast it to a f64.
    /// Returns None otherwise.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Json::I64(n) => Some(n as f64),
            Json::U64(n) => Some(n as f64),
            Json::F64(n) => Some(n),
            _ => None
        }
    }

    /// Returns true if the Json value is a Boolean. Returns false otherwise.
    pub fn is_boolean(&self) -> bool {
        self.as_boolean().is_some()
    }

    /// If the Json value is a Boolean, returns the associated bool.
    /// Returns None otherwise.
    pub fn as_boolean(&self) -> Option<bool> {
        match *self {
            Json::Boolean(b) => Some(b),
            _ => None
        }
    }

    /// Returns true if the Json value is a Null. Returns false otherwise.
    pub fn is_null(&self) -> bool {
        self.as_null().is_some()
    }

    /// If the Json value is a Null, returns ().
    /// Returns None otherwise.
    pub fn as_null(&self) -> Option<()> {
        match *self {
            Json::Null => Some(()),
            _ => None
        }
    }
}

impl<'a> Index<&'a str>  for Json {
    type Output = Json;

    fn index(&self, idx: &'a str) -> &Json {
        self.find(idx).unwrap()
    }
}

impl Index<usize> for Json {
    type Output = Json;

    fn index<'a>(&'a self, idx: usize) -> &'a Json {
        match *self {
            Json::Array(ref v) => &v[idx],
            _ => panic!("can only index Json with usize if it is an array")
        }
    }
}

/// The output of the streaming parser.
#[derive(PartialEq, Clone, Debug)]
pub enum JsonEvent {
    ObjectStart,
    ObjectEnd,
    ArrayStart,
    ArrayEnd,
    BooleanValue(bool),
    I64Value(i64),
    U64Value(u64),
    F64Value(f64),
    StringValue(string::String),
    NullValue,
    Error(ParserError),
}

#[derive(PartialEq, Debug)]
enum ParserState {
    // Parse a value in an array, true means first element.
    ParseArray(bool),
    // Parse ',' or ']' after an element in an array.
    ParseArrayComma,
    // Parse a key:value in an object, true means first element.
    ParseObject(bool),
    // Parse ',' or ']' after an element in an object.
    ParseObjectComma,
    // Initial state.
    ParseStart,
    // Expecting the stream to end.
    ParseBeforeFinish,
    // Parsing can't continue.
    ParseFinished,
}

/// A Stack represents the current position of the parser in the logical
/// structure of the JSON stream.
/// For example foo.bar[3].x
pub struct Stack {
    stack: Vec<InternalStackElement>,
    str_buffer: Vec<u8>,
}

/// StackElements compose a Stack.
/// For example, StackElement::Key("foo"), StackElement::Key("bar"),
/// StackElement::Index(3) and StackElement::Key("x") are the
/// StackElements compositing the stack that represents foo.bar[3].x
#[derive(PartialEq, Clone, Debug)]
pub enum StackElement<'l> {
    Index(u32),
    Key(&'l str),
}

// Internally, Key elements are stored as indices in a buffer to avoid
// allocating a string for every member of an object.
#[derive(PartialEq, Clone, Debug)]
enum InternalStackElement {
    InternalIndex(u32),
    InternalKey(u16, u16), // start, size
}

impl Stack {
    pub fn new() -> Stack {
        Stack { stack: Vec::new(), str_buffer: Vec::new() }
    }

    /// Returns The number of elements in the Stack.
    pub fn len(&self) -> usize { self.stack.len() }

    /// Returns true if the stack is empty.
    pub fn is_empty(&self) -> bool { self.stack.is_empty() }

    /// Provides access to the StackElement at a given index.
    /// lower indices are at the bottom of the stack while higher indices are
    /// at the top.
    pub fn get<'l>(&'l self, idx: usize) -> StackElement<'l> {
        match self.stack[idx] {
            InternalIndex(i) => StackElement::Index(i),
            InternalKey(start, size) => {
                StackElement::Key(str::from_utf8(
                    &self.str_buffer[start as usize .. start as usize + size as usize])
                        .unwrap())
            }
        }
    }

    /// Compares this stack with an array of StackElements.
    pub fn is_equal_to(&self, rhs: &[StackElement]) -> bool {
        if self.stack.len() != rhs.len() { return false; }
        for (i, r) in rhs.iter().enumerate() {
            if self.get(i) != *r { return false; }
        }
        true
    }

    /// Returns true if the bottom-most elements of this stack are the same as
    /// the ones passed as parameter.
    pub fn starts_with(&self, rhs: &[StackElement]) -> bool {
        if self.stack.len() < rhs.len() { return false; }
        for (i, r) in rhs.iter().enumerate() {
            if self.get(i) != *r { return false; }
        }
        true
    }

    /// Returns true if the top-most elements of this stack are the same as
    /// the ones passed as parameter.
    pub fn ends_with(&self, rhs: &[StackElement]) -> bool {
        if self.stack.len() < rhs.len() { return false; }
        let offset = self.stack.len() - rhs.len();
        for (i, r) in rhs.iter().enumerate() {
            if self.get(i + offset) != *r { return false; }
        }
        true
    }

    /// Returns the top-most element (if any).
    pub fn top<'l>(&'l self) -> Option<StackElement<'l>> {
        match self.stack.last() {
            None => None,
            Some(&InternalIndex(i)) => Some(StackElement::Index(i)),
            Some(&InternalKey(start, size)) => {
                Some(StackElement::Key(str::from_utf8(
                    &self.str_buffer[start as usize .. (start+size) as usize]
                ).unwrap()))
            }
        }
    }

    // Used by Parser to insert StackElement::Key elements at the top of the stack.
    fn push_key(&mut self, key: string::String) {
        self.stack.push(InternalKey(self.str_buffer.len() as u16, key.len() as u16));
        for c in key.as_bytes() {
            self.str_buffer.push(*c);
        }
    }

    // Used by Parser to insert StackElement::Index elements at the top of the stack.
    fn push_index(&mut self, index: u32) {
        self.stack.push(InternalIndex(index));
    }

    // Used by Parser to remove the top-most element of the stack.
    fn pop(&mut self) {
        assert!(!self.is_empty());
        match *self.stack.last().unwrap() {
            InternalKey(_, sz) => {
                let new_size = self.str_buffer.len() - sz as usize;
                self.str_buffer.truncate(new_size);
            }
            InternalIndex(_) => {}
        }
        self.stack.pop();
    }

    // Used by Parser to test whether the top-most element is an index.
    fn last_is_index(&self) -> bool {
        if self.is_empty() { return false; }
        return match *self.stack.last().unwrap() {
            InternalIndex(_) => true,
            _ => false,
        }
    }

    // Used by Parser to increment the index of the top-most element.
    fn bump_index(&mut self) {
        let len = self.stack.len();
        let idx = match *self.stack.last().unwrap() {
            InternalIndex(i) => { i + 1 }
            _ => { panic!(); }
        };
        self.stack[len - 1] = InternalIndex(idx);
    }
}

/// A streaming JSON parser implemented as an iterator of JsonEvent, consuming
/// an iterator of char.
pub struct Parser<T> {
    rdr: T,
    ch: Option<char>,
    line: usize,
    col: usize,
    // We maintain a stack representing where we are in the logical structure
    // of the JSON stream.
    stack: Stack,
    // A state machine is kept to make it possible to interrupt and resume parsing.
    state: ParserState,
}

impl<T: Iterator<Item=char>> Iterator for Parser<T> {
    type Item = JsonEvent;

    fn next(&mut self) -> Option<JsonEvent> {
        if self.state == ParseFinished {
            return None;
        }

        if self.state == ParseBeforeFinish {
            self.parse_whitespace();
            // Make sure there is no trailing characters.
            if self.eof() {
                self.state = ParseFinished;
                return None;
            } else {
                return Some(self.error_event(TrailingCharacters));
            }
        }

        Some(self.parse())
    }
}

impl<T: Iterator<Item=char>> Parser<T> {
    /// Creates the JSON parser.
    pub fn new(rdr: T) -> Parser<T> {
        let mut p = Parser {
            rdr: rdr,
            ch: Some('\x00'),
            line: 1,
            col: 0,
            stack: Stack::new(),
            state: ParseStart,
        };
        p.bump();
        p
    }

    /// Provides access to the current position in the logical structure of the
    /// JSON stream.
    pub fn stack<'l>(&'l self) -> &'l Stack {
        &self.stack
    }

    fn eof(&self) -> bool { self.ch.is_none() }
    fn ch_or_null(&self) -> char { self.ch.unwrap_or('\x00') }
    fn bump(&mut self) {
        self.ch = self.rdr.next();

        if self.ch_is('\n') {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
    }

    fn next_char(&mut self) -> Option<char> {
        self.bump();
        self.ch
    }
    fn ch_is(&self, c: char) -> bool {
        self.ch == Some(c)
    }

    fn error<U>(&self, reason: ErrorCode) -> Result<U, ParserError> {
        Err(SyntaxError(reason, self.line, self.col))
    }

    fn parse_whitespace(&mut self) {
        while self.ch_is(' ') ||
              self.ch_is('\n') ||
              self.ch_is('\t') ||
              self.ch_is('\r') { self.bump(); }
    }

    fn parse_number(&mut self) -> JsonEvent {
        let mut neg = false;

        if self.ch_is('-') {
            self.bump();
            neg = true;
        }

        let res = match self.parse_u64() {
            Ok(res) => res,
            Err(e) => { return Error(e); }
        };

        if self.ch_is('.') || self.ch_is('e') || self.ch_is('E') {
            let mut res = res as f64;

            if self.ch_is('.') {
                res = match self.parse_decimal(res) {
                    Ok(res) => res,
                    Err(e) => { return Error(e); }
                };
            }

            if self.ch_is('e') || self.ch_is('E') {
                res = match self.parse_exponent(res) {
                    Ok(res) => res,
                    Err(e) => { return Error(e); }
                };
            }

            if neg {
                res *= -1.0;
            }

            F64Value(res)
        } else {
            if neg {
                let res = (res as i64).wrapping_neg();

                // Make sure we didn't underflow.
                if res > 0 {
                    Error(SyntaxError(InvalidNumber, self.line, self.col))
                } else {
                    I64Value(res)
                }
            } else {
                U64Value(res)
            }
        }
    }

    fn parse_u64(&mut self) -> Result<u64, ParserError> {
        let mut accum = 0u64;
        let last_accum = 0; // necessary to detect overflow.

        match self.ch_or_null() {
            '0' => {
                self.bump();

                // A leading '0' must be the only digit before the decimal point.
                if let '0' ... '9' = self.ch_or_null() {
                    return self.error(InvalidNumber)
                }
            },
            '1' ... '9' => {
                while !self.eof() {
                    match self.ch_or_null() {
                        c @ '0' ... '9' => {
                            accum = accum.wrapping_mul(10);
                            accum = accum.wrapping_add((c as u64) - ('0' as u64));

                            // Detect overflow by comparing to the last value.
                            if accum <= last_accum { return self.error(InvalidNumber); }

                            self.bump();
                        }
                        _ => break,
                    }
                }
            }
            _ => return self.error(InvalidNumber),
        }

        Ok(accum)
    }

    fn parse_decimal(&mut self, mut res: f64) -> Result<f64, ParserError> {
        self.bump();

        // Make sure a digit follows the decimal place.
        match self.ch_or_null() {
            '0' ... '9' => (),
             _ => return self.error(InvalidNumber)
        }

        let mut dec = 1.0;
        while !self.eof() {
            match self.ch_or_null() {
                c @ '0' ... '9' => {
                    dec /= 10.0;
                    res += (((c as isize) - ('0' as isize)) as f64) * dec;
                    self.bump();
                }
                _ => break,
            }
        }

        Ok(res)
    }

    fn parse_exponent(&mut self, mut res: f64) -> Result<f64, ParserError> {
        self.bump();

        let mut exp = 0;
        let mut neg_exp = false;

        if self.ch_is('+') {
            self.bump();
        } else if self.ch_is('-') {
            self.bump();
            neg_exp = true;
        }

        // Make sure a digit follows the exponent place.
        match self.ch_or_null() {
            '0' ... '9' => (),
            _ => return self.error(InvalidNumber)
        }
        while !self.eof() {
            match self.ch_or_null() {
                c @ '0' ... '9' => {
                    exp *= 10;
                    exp += (c as usize) - ('0' as usize);

                    self.bump();
                }
                _ => break
            }
        }

        let exp = 10_f64.powi(exp as i32);
        if neg_exp {
            res /= exp;
        } else {
            res *= exp;
        }

        Ok(res)
    }

    fn decode_hex_escape(&mut self) -> Result<u16, ParserError> {
        let mut i = 0;
        let mut n = 0;
        while i < 4 && !self.eof() {
            self.bump();
            n = match self.ch_or_null() {
                c @ '0' ... '9' => n * 16 + ((c as u16) - ('0' as u16)),
                'a' | 'A' => n * 16 + 10,
                'b' | 'B' => n * 16 + 11,
                'c' | 'C' => n * 16 + 12,
                'd' | 'D' => n * 16 + 13,
                'e' | 'E' => n * 16 + 14,
                'f' | 'F' => n * 16 + 15,
                _ => return self.error(InvalidEscape)
            };

            i += 1;
        }

        // Error out if we didn't parse 4 digits.
        if i != 4 {
            return self.error(InvalidEscape);
        }

        Ok(n)
    }

    fn parse_str(&mut self) -> Result<string::String, ParserError> {
        let mut escape = false;
        let mut res = string::String::new();

        loop {
            self.bump();
            if self.eof() {
                return self.error(EOFWhileParsingString);
            }

            if escape {
                match self.ch_or_null() {
                    '"' => res.push('"'),
                    '\\' => res.push('\\'),
                    '/' => res.push('/'),
                    'b' => res.push('\x08'),
                    'f' => res.push('\x0c'),
                    'n' => res.push('\n'),
                    'r' => res.push('\r'),
                    't' => res.push('\t'),
                    'u' => match try!(self.decode_hex_escape()) {
                        0xDC00 ... 0xDFFF => {
                            return self.error(LoneLeadingSurrogateInHexEscape)
                        }

                        // Non-BMP characters are encoded as a sequence of
                        // two hex escapes, representing UTF-16 surrogates.
                        n1 @ 0xD800 ... 0xDBFF => {
                            match (self.next_char(), self.next_char()) {
                                (Some('\\'), Some('u')) => (),
                                _ => return self.error(UnexpectedEndOfHexEscape),
                            }

                            let n2 = try!(self.decode_hex_escape());
                            if n2 < 0xDC00 || n2 > 0xDFFF {
                                return self.error(LoneLeadingSurrogateInHexEscape)
                            }
                            let c = (((n1 - 0xD800) as u32) << 10 |
                                     (n2 - 0xDC00) as u32) + 0x1_0000;
                            res.push(char::from_u32(c).unwrap());
                        }

                        n => match char::from_u32(n as u32) {
                            Some(c) => res.push(c),
                            None => return self.error(InvalidUnicodeCodePoint),
                        },
                    },
                    _ => return self.error(InvalidEscape),
                }
                escape = false;
            } else if self.ch_is('\\') {
                escape = true;
            } else {
                match self.ch {
                    Some('"') => {
                        self.bump();
                        return Ok(res);
                    },
                    Some(c) => res.push(c),
                    None => unreachable!()
                }
            }
        }
    }

    // Invoked at each iteration, consumes the stream until it has enough
    // information to return a JsonEvent.
    // Manages an internal state so that parsing can be interrupted and resumed.
    // Also keeps track of the position in the logical structure of the json
    // stream isize the form of a stack that can be queried by the user using the
    // stack() method.
    fn parse(&mut self) -> JsonEvent {
        loop {
            // The only paths where the loop can spin a new iteration
            // are in the cases ParseArrayComma and ParseObjectComma if ','
            // is parsed. In these cases the state is set to (respectively)
            // ParseArray(false) and ParseObject(false), which always return,
            // so there is no risk of getting stuck in an infinite loop.
            // All other paths return before the end of the loop's iteration.
            self.parse_whitespace();

            match self.state {
                ParseStart => {
                    return self.parse_start();
                }
                ParseArray(first) => {
                    return self.parse_array(first);
                }
                ParseArrayComma => {
                    match self.parse_array_comma_or_end() {
                        Some(evt) => { return evt; }
                        None => {}
                    }
                }
                ParseObject(first) => {
                    return self.parse_object(first);
                }
                ParseObjectComma => {
                    self.stack.pop();
                    if self.ch_is(',') {
                        self.state = ParseObject(false);
                        self.bump();
                    } else {
                        return self.parse_object_end();
                    }
                }
                _ => {
                    return self.error_event(InvalidSyntax);
                }
            }
        }
    }

    fn parse_start(&mut self) -> JsonEvent {
        let val = self.parse_value();
        self.state = match val {
            Error(_) => ParseFinished,
            ArrayStart => ParseArray(true),
            ObjectStart => ParseObject(true),
            _ => ParseBeforeFinish,
        };
        val
    }

    fn parse_array(&mut self, first: bool) -> JsonEvent {
        if self.ch_is(']') {
            if !first {
                self.error_event(InvalidSyntax)
            } else {
                self.state = if self.stack.is_empty() {
                    ParseBeforeFinish
                } else if self.stack.last_is_index() {
                    ParseArrayComma
                } else {
                    ParseObjectComma
                };
                self.bump();
                ArrayEnd
            }
        } else {
            if first {
                self.stack.push_index(0);
            }
            let val = self.parse_value();
            self.state = match val {
                Error(_) => ParseFinished,
                ArrayStart => ParseArray(true),
                ObjectStart => ParseObject(true),
                _ => ParseArrayComma,
            };
            val
        }
    }

    fn parse_array_comma_or_end(&mut self) -> Option<JsonEvent> {
        if self.ch_is(',') {
            self.stack.bump_index();
            self.state = ParseArray(false);
            self.bump();
            None
        } else if self.ch_is(']') {
            self.stack.pop();
            self.state = if self.stack.is_empty() {
                ParseBeforeFinish
            } else if self.stack.last_is_index() {
                ParseArrayComma
            } else {
                ParseObjectComma
            };
            self.bump();
            Some(ArrayEnd)
        } else if self.eof() {
            Some(self.error_event(EOFWhileParsingArray))
        } else {
            Some(self.error_event(InvalidSyntax))
        }
    }

    fn parse_object(&mut self, first: bool) -> JsonEvent {
        if self.ch_is('}') {
            if !first {
                if self.stack.is_empty() {
                    return self.error_event(TrailingComma);
                } else {
                    self.stack.pop();
                }
            }
            self.state = if self.stack.is_empty() {
                ParseBeforeFinish
            } else if self.stack.last_is_index() {
                ParseArrayComma
            } else {
                ParseObjectComma
            };
            self.bump();
            return ObjectEnd;
        }
        if self.eof() {
            return self.error_event(EOFWhileParsingObject);
        }
        if !self.ch_is('"') {
            return self.error_event(KeyMustBeAString);
        }
        let s = match self.parse_str() {
            Ok(s) => s,
            Err(e) => {
                self.state = ParseFinished;
                return Error(e);
            }
        };
        self.parse_whitespace();
        if self.eof() {
            return self.error_event(EOFWhileParsingObject);
        } else if self.ch_or_null() != ':' {
            return self.error_event(ExpectedColon);
        }
        self.stack.push_key(s);
        self.bump();
        self.parse_whitespace();

        let val = self.parse_value();

        self.state = match val {
            Error(_) => ParseFinished,
            ArrayStart => ParseArray(true),
            ObjectStart => ParseObject(true),
            _ => ParseObjectComma,
        };
        val
    }

    fn parse_object_end(&mut self) -> JsonEvent {
        if self.ch_is('}') {
            self.state = if self.stack.is_empty() {
                ParseBeforeFinish
            } else if self.stack.last_is_index() {
                ParseArrayComma
            } else {
                ParseObjectComma
            };
            self.bump();
            ObjectEnd
        } else if self.eof() {
            self.error_event(EOFWhileParsingObject)
        } else {
            self.error_event(InvalidSyntax)
        }
    }

    fn parse_value(&mut self) -> JsonEvent {
        if self.eof() { return self.error_event(EOFWhileParsingValue); }
        match self.ch_or_null() {
            'n' => { self.parse_ident("ull", NullValue) }
            't' => { self.parse_ident("rue", BooleanValue(true)) }
            'f' => { self.parse_ident("alse", BooleanValue(false)) }
            '0' ... '9' | '-' => self.parse_number(),
            '"' => match self.parse_str() {
                Ok(s) => StringValue(s),
                Err(e) => Error(e),
            },
            '[' => {
                self.bump();
                ArrayStart
            }
            '{' => {
                self.bump();
                ObjectStart
            }
            _ => { self.error_event(InvalidSyntax) }
        }
    }

    fn parse_ident(&mut self, ident: &str, value: JsonEvent) -> JsonEvent {
        if ident.chars().all(|c| Some(c) == self.next_char()) {
            self.bump();
            value
        } else {
            Error(SyntaxError(InvalidSyntax, self.line, self.col))
        }
    }

    fn error_event(&mut self, reason: ErrorCode) -> JsonEvent {
        self.state = ParseFinished;
        Error(SyntaxError(reason, self.line, self.col))
    }
}

/// A Builder consumes a json::Parser to create a generic Json structure.
pub struct Builder<T> {
    parser: Parser<T>,
    token: Option<JsonEvent>,
}

impl<T: Iterator<Item=char>> Builder<T> {
    /// Create a JSON Builder.
    pub fn new(src: T) -> Builder<T> {
        Builder { parser: Parser::new(src), token: None, }
    }

    // Decode a Json value from a Parser.
    pub fn build(&mut self) -> Result<Json, BuilderError> {
        self.bump();
        let result = self.build_value();
        self.bump();
        match self.token {
            None => {}
            Some(Error(ref e)) => { return Err(e.clone()); }
            ref tok => { panic!("unexpected token {:?}", tok.clone()); }
        }
        result
    }

    fn bump(&mut self) {
        self.token = self.parser.next();
    }

    fn build_value(&mut self) -> Result<Json, BuilderError> {
        match self.token {
            Some(NullValue) => Ok(Json::Null),
            Some(I64Value(n)) => Ok(Json::I64(n)),
            Some(U64Value(n)) => Ok(Json::U64(n)),
            Some(F64Value(n)) => Ok(Json::F64(n)),
            Some(BooleanValue(b)) => Ok(Json::Boolean(b)),
            Some(StringValue(ref mut s)) => {
                let mut temp = string::String::new();
                swap(s, &mut temp);
                Ok(Json::String(temp))
            }
            Some(Error(ref e)) => Err(e.clone()),
            Some(ArrayStart) => self.build_array(),
            Some(ObjectStart) => self.build_object(),
            Some(ObjectEnd) => self.parser.error(InvalidSyntax),
            Some(ArrayEnd) => self.parser.error(InvalidSyntax),
            None => self.parser.error(EOFWhileParsingValue),
        }
    }

    fn build_array(&mut self) -> Result<Json, BuilderError> {
        self.bump();
        let mut values = Vec::new();

        loop {
            if self.token == Some(ArrayEnd) {
                return Ok(Json::Array(values.into_iter().collect()));
            }
            match self.build_value() {
                Ok(v) => values.push(v),
                Err(e) => { return Err(e) }
            }
            self.bump();
        }
    }

    fn build_object(&mut self) -> Result<Json, BuilderError> {
        self.bump();

        let mut values = BTreeMap::new();

        loop {
            match self.token {
                Some(ObjectEnd) => { return Ok(Json::Object(values)); }
                Some(Error(ref e)) => { return Err(e.clone()); }
                None => { break; }
                _ => {}
            }
            let key = match self.parser.stack().top() {
                Some(StackElement::Key(k)) => { k.to_owned() }
                _ => { panic!("invalid state"); }
            };
            match self.build_value() {
                Ok(value) => { values.insert(key, value); }
                Err(e) => { return Err(e); }
            }
            self.bump();
        }
        self.parser.error(EOFWhileParsingObject)
    }
}

/// Decodes a json value from an `&mut io::Read`
pub fn from_reader(rdr: &mut Read) -> Result<Json, BuilderError> {
    let mut contents = Vec::new();
    match rdr.read_to_end(&mut contents) {
        Ok(c)  => c,
        Err(e) => return Err(io_error_to_error(e))
    };
    let s = match str::from_utf8(&contents).ok() {
        Some(s) => s,
        _       => return Err(SyntaxError(NotUtf8, 0, 0))
    };
    let mut builder = Builder::new(s.chars());
    builder.build()
}

/// Decodes a json value from a string
pub fn from_str(s: &str) -> Result<Json, BuilderError> {
    let mut builder = Builder::new(s.chars());
    builder.build()
}

/// A structure to decode JSON to values in rust.
pub struct Decoder {
    stack: Vec<Json>,
}

impl Decoder {
    /// Creates a new decoder instance for decoding the specified JSON value.
    pub fn new(json: Json) -> Decoder {
        Decoder { stack: vec![json] }
    }
}

impl Decoder {
    fn pop(&mut self) -> Json {
        self.stack.pop().unwrap()
    }
}

macro_rules! expect {
    ($e:expr, Null) => ({
        match $e {
            Json::Null => Ok(()),
            other => Err(ExpectedError("Null".to_owned(),
                                       format!("{}", other)))
        }
    });
    ($e:expr, $t:ident) => ({
        match $e {
            Json::$t(v) => Ok(v),
            other => {
                Err(ExpectedError(stringify!($t).to_owned(),
                                  format!("{}", other)))
            }
        }
    })
}

macro_rules! read_primitive {
    ($name:ident, $ty:ty) => {
        fn $name(&mut self) -> DecodeResult<$ty> {
            match self.pop() {
                Json::I64(f) => Ok(f as $ty),
                Json::U64(f) => Ok(f as $ty),
                Json::F64(f) => Err(ExpectedError("Integer".to_owned(), format!("{}", f))),
                // re: #12967.. a type w/ numeric keys (ie HashMap<usize, V> etc)
                // is going to have a string here, as per JSON spec.
                Json::String(s) => match s.parse().ok() {
                    Some(f) => Ok(f),
                    None => Err(ExpectedError("Number".to_owned(), s)),
                },
                value => Err(ExpectedError("Number".to_owned(), format!("{}", value))),
            }
        }
    }
}

impl ::Decoder for Decoder {
    type Error = DecoderError;

    fn read_nil(&mut self) -> DecodeResult<()> {
        expect!(self.pop(), Null)
    }

    read_primitive! { read_uint, usize }
    read_primitive! { read_u8, u8 }
    read_primitive! { read_u16, u16 }
    read_primitive! { read_u32, u32 }
    read_primitive! { read_u64, u64 }
    read_primitive! { read_int, isize }
    read_primitive! { read_i8, i8 }
    read_primitive! { read_i16, i16 }
    read_primitive! { read_i32, i32 }
    read_primitive! { read_i64, i64 }

    fn read_f32(&mut self) -> DecodeResult<f32> { self.read_f64().map(|x| x as f32) }

    fn read_f64(&mut self) -> DecodeResult<f64> {
        match self.pop() {
            Json::I64(f) => Ok(f as f64),
            Json::U64(f) => Ok(f as f64),
            Json::F64(f) => Ok(f),
            Json::String(s) => {
                // re: #12967.. a type w/ numeric keys (ie HashMap<usize, V> etc)
                // is going to have a string here, as per JSON spec.
                match s.parse().ok() {
                    Some(f) => Ok(f),
                    None => Err(ExpectedError("Number".to_owned(), s)),
                }
            },
            Json::Null => Ok(f64::NAN),
            value => Err(ExpectedError("Number".to_owned(), format!("{}", value)))
        }
    }

    fn read_bool(&mut self) -> DecodeResult<bool> {
        expect!(self.pop(), Boolean)
    }

    fn read_char(&mut self) -> DecodeResult<char> {
        let s = try!(self.read_str());
        {
            let mut it = s.chars();
            match (it.next(), it.next()) {
                // exactly one character
                (Some(c), None) => return Ok(c),
                _ => ()
            }
        }
        Err(ExpectedError("single character string".to_owned(), format!("{}", s)))
    }

    fn read_str(&mut self) -> DecodeResult<string::String> {
        expect!(self.pop(), String)
    }

    fn read_enum<T, F>(&mut self, _name: &str, f: F) -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        f(self)
    }

    fn read_enum_variant<T, F>(&mut self, names: &[&str],
                               mut f: F) -> DecodeResult<T>
        where F: FnMut(&mut Decoder, usize) -> DecodeResult<T>,
    {
        let name = match self.pop() {
            Json::String(s) => s,
            Json::Object(mut o) => {
                let n = match o.remove(&"variant".to_owned()) {
                    Some(Json::String(s)) => s,
                    Some(val) => {
                        return Err(ExpectedError("String".to_owned(), format!("{}", val)))
                    }
                    None => {
                        return Err(MissingFieldError("variant".to_owned()))
                    }
                };
                match o.remove(&"fields".to_string()) {
                    Some(Json::Array(l)) => {
                        for field in l.into_iter().rev() {
                            self.stack.push(field);
                        }
                    },
                    Some(val) => {
                        return Err(ExpectedError("Array".to_owned(), format!("{}", val)))
                    }
                    None => {
                        return Err(MissingFieldError("fields".to_owned()))
                    }
                }
                n
            }
            json => {
                return Err(ExpectedError("String or Object".to_owned(), format!("{}", json)))
            }
        };
        let idx = match names.iter().position(|n| *n == &name[..]) {
            Some(idx) => idx,
            None => return Err(UnknownVariantError(name))
        };
        f(self, idx)
    }

    fn read_enum_variant_arg<T, F>(&mut self, _idx: usize, f: F) -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        f(self)
    }

    fn read_enum_struct_variant<T, F>(&mut self, names: &[&str], f: F) -> DecodeResult<T> where
        F: FnMut(&mut Decoder, usize) -> DecodeResult<T>,
    {
        self.read_enum_variant(names, f)
    }


    fn read_enum_struct_variant_field<T, F>(&mut self,
                                         _name: &str,
                                         idx: usize,
                                         f: F)
                                         -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        self.read_enum_variant_arg(idx, f)
    }

    fn read_struct<T, F>(&mut self, _name: &str, _len: usize, f: F) -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        let value = try!(f(self));
        self.pop();
        Ok(value)
    }

    fn read_struct_field<T, F>(&mut self,
                               name: &str,
                               _idx: usize,
                               f: F)
                               -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        let mut obj = try!(expect!(self.pop(), Object));

        let value = match obj.remove(&name.to_string()) {
            None => {
                // Add a Null and try to parse it as an Option<_>
                // to get None as a default value.
                self.stack.push(Json::Null);
                match f(self) {
                    Ok(x) => x,
                    Err(_) => return Err(MissingFieldError(name.to_string())),
                }
            },
            Some(json) => {
                self.stack.push(json);
                try!(f(self))
            }
        };
        self.stack.push(Json::Object(obj));
        Ok(value)
    }

    fn read_tuple<T, F>(&mut self, tuple_len: usize, f: F) -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        self.read_seq(move |d, len| {
            if len == tuple_len {
                f(d)
            } else {
                Err(ExpectedError(format!("Tuple{}", tuple_len), format!("Tuple{}", len)))
            }
        })
    }

    fn read_tuple_arg<T, F>(&mut self, idx: usize, f: F) -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        self.read_seq_elt(idx, f)
    }

    fn read_tuple_struct<T, F>(&mut self,
                               _name: &str,
                               len: usize,
                               f: F)
                               -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        self.read_tuple(len, f)
    }

    fn read_tuple_struct_arg<T, F>(&mut self,
                                   idx: usize,
                                   f: F)
                                   -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        self.read_tuple_arg(idx, f)
    }

    fn read_option<T, F>(&mut self, mut f: F) -> DecodeResult<T> where
        F: FnMut(&mut Decoder, bool) -> DecodeResult<T>,
    {
        match self.pop() {
            Json::Null => f(self, false),
            value => { self.stack.push(value); f(self, true) }
        }
    }

    fn read_seq<T, F>(&mut self, f: F) -> DecodeResult<T> where
        F: FnOnce(&mut Decoder, usize) -> DecodeResult<T>,
    {
        let array = try!(expect!(self.pop(), Array));
        let len = array.len();
        for v in array.into_iter().rev() {
            self.stack.push(v);
        }
        f(self, len)
    }

    fn read_seq_elt<T, F>(&mut self, _idx: usize, f: F) -> DecodeResult<T> where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        f(self)
    }

    fn read_map<T, F>(&mut self, f: F) -> DecodeResult<T> where
        F: FnOnce(&mut Decoder, usize) -> DecodeResult<T>,
    {
        let obj = try!(expect!(self.pop(), Object));
        let len = obj.len();
        for (key, value) in obj {
            self.stack.push(value);
            self.stack.push(Json::String(key));
        }
        f(self, len)
    }

    fn read_map_elt_key<T, F>(&mut self, _idx: usize, f: F) -> DecodeResult<T> where
       F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        f(self)
    }

    fn read_map_elt_val<T, F>(&mut self, _idx: usize, f: F) -> DecodeResult<T> where
       F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        f(self)
    }

    fn error(&mut self, err: &str) -> DecoderError {
        ApplicationError(err.to_string())
    }
}

/// A trait for converting values to JSON
pub trait ToJson {
    /// Converts the value of `self` to an instance of JSON
    fn to_json(&self) -> Json;
}

macro_rules! to_json_impl_i64 {
    ($($t:ty), +) => (
        $(impl ToJson for $t {
            fn to_json(&self) -> Json {
                Json::I64(*self as i64)
            }
        })+
    )
}

to_json_impl_i64! { isize, i8, i16, i32, i64 }

macro_rules! to_json_impl_u64 {
    ($($t:ty), +) => (
        $(impl ToJson for $t {
            fn to_json(&self) -> Json {
                Json::U64(*self as u64)
            }
        })+
    )
}

to_json_impl_u64! { usize, u8, u16, u32, u64 }

impl ToJson for Json {
    fn to_json(&self) -> Json { self.clone() }
}

impl ToJson for f32 {
    fn to_json(&self) -> Json { (*self as f64).to_json() }
}

impl ToJson for f64 {
    fn to_json(&self) -> Json {
        match self.classify() {
            Fp::Nan | Fp::Infinite => Json::Null,
            _                  => Json::F64(*self)
        }
    }
}

impl ToJson for () {
    fn to_json(&self) -> Json { Json::Null }
}

impl ToJson for bool {
    fn to_json(&self) -> Json { Json::Boolean(*self) }
}

impl ToJson for str {
    fn to_json(&self) -> Json { Json::String(self.to_string()) }
}

impl ToJson for string::String {
    fn to_json(&self) -> Json { Json::String((*self).clone()) }
}

macro_rules! tuple_impl {
    // use variables to indicate the arity of the tuple
    ($($tyvar:ident),* ) => {
        // the trailing commas are for the 1 tuple
        impl<
            $( $tyvar : ToJson ),*
            > ToJson for ( $( $tyvar ),* , ) {

            #[inline]
            #[allow(non_snake_case)]
            fn to_json(&self) -> Json {
                match *self {
                    ($(ref $tyvar),*,) => Json::Array(vec![$($tyvar.to_json()),*])
                }
            }
        }
    }
}

tuple_impl!{A}
tuple_impl!{A, B}
tuple_impl!{A, B, C}
tuple_impl!{A, B, C, D}
tuple_impl!{A, B, C, D, E}
tuple_impl!{A, B, C, D, E, F}
tuple_impl!{A, B, C, D, E, F, G}
tuple_impl!{A, B, C, D, E, F, G, H}
tuple_impl!{A, B, C, D, E, F, G, H, I}
tuple_impl!{A, B, C, D, E, F, G, H, I, J}
tuple_impl!{A, B, C, D, E, F, G, H, I, J, K}
tuple_impl!{A, B, C, D, E, F, G, H, I, J, K, L}

impl<A: ToJson> ToJson for [A] {
    fn to_json(&self) -> Json { Json::Array(self.iter().map(|elt| elt.to_json()).collect()) }
}

impl<A: ToJson> ToJson for Vec<A> {
    fn to_json(&self) -> Json { Json::Array(self.iter().map(|elt| elt.to_json()).collect()) }
}

impl<A: ToJson> ToJson for BTreeMap<string::String, A> {
    fn to_json(&self) -> Json {
        let mut d = BTreeMap::new();
        for (key, value) in self {
            d.insert((*key).clone(), value.to_json());
        }
        Json::Object(d)
    }
}

impl<A: ToJson> ToJson for HashMap<string::String, A> {
    fn to_json(&self) -> Json {
        let mut d = BTreeMap::new();
        for (key, value) in self {
            d.insert((*key).clone(), value.to_json());
        }
        Json::Object(d)
    }
}

impl<A:ToJson> ToJson for Option<A> {
    fn to_json(&self) -> Json {
        match *self {
            None => Json::Null,
            Some(ref value) => value.to_json()
        }
    }
}

struct FormatShim<'a, 'b: 'a> {
    inner: &'a mut fmt::Formatter<'b>,
}

impl<'a, 'b> fmt::Write for FormatShim<'a, 'b> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        match self.inner.write_str(s) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error)
        }
    }
}

impl fmt::Display for Json {
    /// Encodes a json value into a string
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut shim = FormatShim { inner: f };
        let mut encoder = Encoder::new(&mut shim);
        match self.encode(&mut encoder) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error)
        }
    }
}

impl<'a> fmt::Display for PrettyJson<'a> {
    /// Encodes a json value into a string
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut shim = FormatShim { inner: f };
        let mut encoder = PrettyEncoder::new(&mut shim);
        match self.inner.encode(&mut encoder) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error)
        }
    }
}

impl<'a, T: Encodable> fmt::Display for AsJson<'a, T> {
    /// Encodes a json value into a string
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut shim = FormatShim { inner: f };
        let mut encoder = Encoder::new(&mut shim);
        match self.inner.encode(&mut encoder) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error)
        }
    }
}

impl<'a, T> AsPrettyJson<'a, T> {
    /// Set the indentation level for the emitted JSON
    pub fn indent(mut self, indent: usize) -> AsPrettyJson<'a, T> {
        self.indent = Some(indent);
        self
    }
}

impl<'a, T: Encodable> fmt::Display for AsPrettyJson<'a, T> {
    /// Encodes a json value into a string
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut shim = FormatShim { inner: f };
        let mut encoder = PrettyEncoder::new(&mut shim);
        match self.indent {
            Some(n) => encoder.set_indent(n),
            None => {}
        }
        match self.inner.encode(&mut encoder) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error)
        }
    }
}

impl FromStr for Json {
    type Err = BuilderError;
    fn from_str(s: &str) -> Result<Json, BuilderError> {
        from_str(s)
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::Animal::*;
    use self::DecodeEnum::*;
    use self::test::Bencher;
    use {Encodable, Decodable};
    use super::Json::*;
    use super::ErrorCode::*;
    use super::ParserError::*;
    use super::DecoderError::*;
    use super::JsonEvent::*;
    use super::{Json, from_str, DecodeResult, DecoderError, JsonEvent, Parser,
                StackElement, Stack, Decoder, Encoder, EncoderError};
    use std::{i64, u64, f32, f64};
    use std::io::prelude::*;
    use std::collections::BTreeMap;
    use std::string;

    #[derive(RustcDecodable, Eq, PartialEq, Debug)]
    struct OptionData {
        opt: Option<usize>,
    }

    #[test]
    fn test_decode_option_none() {
        let s ="{}";
        let obj: OptionData = super::decode(s).unwrap();
        assert_eq!(obj, OptionData { opt: None });
    }

    #[test]
    fn test_decode_option_some() {
        let s = "{ \"opt\": 10 }";
        let obj: OptionData = super::decode(s).unwrap();
        assert_eq!(obj, OptionData { opt: Some(10) });
    }

    #[test]
    fn test_decode_option_malformed() {
        check_err::<OptionData>("{ \"opt\": [] }",
                                ExpectedError("Number".to_string(), "[]".to_string()));
        check_err::<OptionData>("{ \"opt\": false }",
                                ExpectedError("Number".to_string(), "false".to_string()));
    }

    #[derive(PartialEq, RustcEncodable, RustcDecodable, Debug)]
    enum Animal {
        Dog,
        Frog(string::String, isize)
    }

    #[derive(PartialEq, RustcEncodable, RustcDecodable, Debug)]
    struct Inner {
        a: (),
        b: usize,
        c: Vec<string::String>,
    }

    #[derive(PartialEq, RustcEncodable, RustcDecodable, Debug)]
    struct Outer {
        inner: Vec<Inner>,
    }

    fn mk_object(items: &[(string::String, Json)]) -> Json {
        let mut d = BTreeMap::new();

        for item in items {
            match *item {
                (ref key, ref value) => { d.insert((*key).clone(), (*value).clone()); },
            }
        };

        Object(d)
    }

    #[test]
    fn test_from_str_trait() {
        let s = "null";
        assert!(s.parse::<Json>().unwrap() == s.parse().unwrap());
    }

    #[test]
    fn test_write_null() {
        assert_eq!(Null.to_string(), "null");
        assert_eq!(Null.pretty().to_string(), "null");
    }

    #[test]
    fn test_write_i64() {
        assert_eq!(U64(0).to_string(), "0");
        assert_eq!(U64(0).pretty().to_string(), "0");

        assert_eq!(U64(1234).to_string(), "1234");
        assert_eq!(U64(1234).pretty().to_string(), "1234");

        assert_eq!(I64(-5678).to_string(), "-5678");
        assert_eq!(I64(-5678).pretty().to_string(), "-5678");

        assert_eq!(U64(7650007200025252000).to_string(), "7650007200025252000");
        assert_eq!(U64(7650007200025252000).pretty().to_string(), "7650007200025252000");
    }

    #[test]
    fn test_write_f64() {
        assert_eq!(F64(3.0).to_string(), "3.0");
        assert_eq!(F64(3.0).pretty().to_string(), "3.0");

        assert_eq!(F64(3.1).to_string(), "3.1");
        assert_eq!(F64(3.1).pretty().to_string(), "3.1");

        assert_eq!(F64(-1.5).to_string(), "-1.5");
        assert_eq!(F64(-1.5).pretty().to_string(), "-1.5");

        assert_eq!(F64(0.5).to_string(), "0.5");
        assert_eq!(F64(0.5).pretty().to_string(), "0.5");

        assert_eq!(F64(f64::NAN).to_string(), "null");
        assert_eq!(F64(f64::NAN).pretty().to_string(), "null");

        assert_eq!(F64(f64::INFINITY).to_string(), "null");
        assert_eq!(F64(f64::INFINITY).pretty().to_string(), "null");

        assert_eq!(F64(f64::NEG_INFINITY).to_string(), "null");
        assert_eq!(F64(f64::NEG_INFINITY).pretty().to_string(), "null");
    }

    #[test]
    fn test_write_str() {
        assert_eq!(String("".to_string()).to_string(), "\"\"");
        assert_eq!(String("".to_string()).pretty().to_string(), "\"\"");

        assert_eq!(String("homura".to_string()).to_string(), "\"homura\"");
        assert_eq!(String("madoka".to_string()).pretty().to_string(), "\"madoka\"");
    }

    #[test]
    fn test_write_bool() {
        assert_eq!(Boolean(true).to_string(), "true");
        assert_eq!(Boolean(true).pretty().to_string(), "true");

        assert_eq!(Boolean(false).to_string(), "false");
        assert_eq!(Boolean(false).pretty().to_string(), "false");
    }

    #[test]
    fn test_write_array() {
        assert_eq!(Array(vec![]).to_string(), "[]");
        assert_eq!(Array(vec![]).pretty().to_string(), "[]");

        assert_eq!(Array(vec![Boolean(true)]).to_string(), "[true]");
        assert_eq!(
            Array(vec![Boolean(true)]).pretty().to_string(),
            "\
            [\n  \
                true\n\
            ]"
        );

        let long_test_array = Array(vec![
            Boolean(false),
            Null,
            Array(vec![String("foo\nbar".to_string()), F64(3.5)])]);

        assert_eq!(long_test_array.to_string(),
            "[false,null,[\"foo\\nbar\",3.5]]");
        assert_eq!(
            long_test_array.pretty().to_string(),
            "\
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
        assert_eq!(mk_object(&[]).to_string(), "{}");
        assert_eq!(mk_object(&[]).pretty().to_string(), "{}");

        assert_eq!(
            mk_object(&[
                ("a".to_string(), Boolean(true))
            ]).to_string(),
            "{\"a\":true}"
        );
        assert_eq!(
            mk_object(&[("a".to_string(), Boolean(true))]).pretty().to_string(),
            "\
            {\n  \
                \"a\": true\n\
            }"
        );

        let complex_obj = mk_object(&[
                ("b".to_string(), Array(vec![
                    mk_object(&[("c".to_string(), String("\x0c\r".to_string()))]),
                    mk_object(&[("d".to_string(), String("".to_string()))])
                ]))
            ]);

        assert_eq!(
            complex_obj.to_string(),
            "{\
                \"b\":[\
                    {\"c\":\"\\f\\r\"},\
                    {\"d\":\"\"}\
                ]\
            }"
        );
        assert_eq!(
            complex_obj.pretty().to_string(),
            "\
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

        let a = mk_object(&[
            ("a".to_string(), Boolean(true)),
            ("b".to_string(), Array(vec![
                mk_object(&[("c".to_string(), String("\x0c\r".to_string()))]),
                mk_object(&[("d".to_string(), String("".to_string()))])
            ]))
        ]);

        // We can't compare the strings directly because the object fields be
        // printed in a different order.
        assert_eq!(a.clone(), a.to_string().parse().unwrap());
        assert_eq!(a.clone(), a.pretty().to_string().parse().unwrap());
    }

    #[test]
    fn test_write_enum() {
        let animal = Dog;
        assert_eq!(
            format!("{}", super::as_json(&animal)),
            "\"Dog\""
        );
        assert_eq!(
            format!("{}", super::as_pretty_json(&animal)),
            "\"Dog\""
        );

        let animal = Frog("Henry".to_string(), 349);
        assert_eq!(
            format!("{}", super::as_json(&animal)),
            "{\"variant\":\"Frog\",\"fields\":[\"Henry\",349]}"
        );
        assert_eq!(
            format!("{}", super::as_pretty_json(&animal)),
            "{\n  \
               \"variant\": \"Frog\",\n  \
               \"fields\": [\n    \
                 \"Henry\",\n    \
                 349\n  \
               ]\n\
             }"
        );
    }

    macro_rules! check_encoder_for_simple {
        ($value:expr, $expected:expr) => ({
            let s = format!("{}", super::as_json(&$value));
            assert_eq!(s, $expected);

            let s = format!("{}", super::as_pretty_json(&$value));
            assert_eq!(s, $expected);
        })
    }

    #[test]
    fn test_write_some() {
        check_encoder_for_simple!(Some("jodhpurs".to_string()), "\"jodhpurs\"");
    }

    #[test]
    fn test_write_none() {
        check_encoder_for_simple!(None::<string::String>, "null");
    }

    #[test]
    fn test_write_char() {
        check_encoder_for_simple!('a', "\"a\"");
        check_encoder_for_simple!('\t', "\"\\t\"");
        check_encoder_for_simple!('\u{0000}', "\"\\u0000\"");
        check_encoder_for_simple!('\u{001b}', "\"\\u001b\"");
        check_encoder_for_simple!('\u{007f}', "\"\\u007f\"");
        check_encoder_for_simple!('\u{00a0}', "\"\u{00a0}\"");
        check_encoder_for_simple!('\u{abcd}', "\"\u{abcd}\"");
        check_encoder_for_simple!('\u{10ffff}', "\"\u{10ffff}\"");
    }

    #[test]
    fn test_trailing_characters() {
        assert_eq!(from_str("nulla"),  Err(SyntaxError(TrailingCharacters, 1, 5)));
        assert_eq!(from_str("truea"),  Err(SyntaxError(TrailingCharacters, 1, 5)));
        assert_eq!(from_str("falsea"), Err(SyntaxError(TrailingCharacters, 1, 6)));
        assert_eq!(from_str("1a"),     Err(SyntaxError(TrailingCharacters, 1, 2)));
        assert_eq!(from_str("[]a"),    Err(SyntaxError(TrailingCharacters, 1, 3)));
        assert_eq!(from_str("{}a"),    Err(SyntaxError(TrailingCharacters, 1, 3)));
    }

    #[test]
    fn test_read_identifiers() {
        assert_eq!(from_str("n"),    Err(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(from_str("nul"),  Err(SyntaxError(InvalidSyntax, 1, 4)));
        assert_eq!(from_str("t"),    Err(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(from_str("truz"), Err(SyntaxError(InvalidSyntax, 1, 4)));
        assert_eq!(from_str("f"),    Err(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(from_str("faz"),  Err(SyntaxError(InvalidSyntax, 1, 3)));

        assert_eq!(from_str("null"), Ok(Null));
        assert_eq!(from_str("true"), Ok(Boolean(true)));
        assert_eq!(from_str("false"), Ok(Boolean(false)));
        assert_eq!(from_str(" null "), Ok(Null));
        assert_eq!(from_str(" true "), Ok(Boolean(true)));
        assert_eq!(from_str(" false "), Ok(Boolean(false)));
    }

    #[test]
    fn test_decode_identifiers() {
        let v: () = super::decode("null").unwrap();
        assert_eq!(v, ());

        let v: bool = super::decode("true").unwrap();
        assert_eq!(v, true);

        let v: bool = super::decode("false").unwrap();
        assert_eq!(v, false);
    }

    #[test]
    fn test_read_number() {
        assert_eq!(from_str("+"),   Err(SyntaxError(InvalidSyntax, 1, 1)));
        assert_eq!(from_str("."),   Err(SyntaxError(InvalidSyntax, 1, 1)));
        assert_eq!(from_str("NaN"), Err(SyntaxError(InvalidSyntax, 1, 1)));
        assert_eq!(from_str("-"),   Err(SyntaxError(InvalidNumber, 1, 2)));
        assert_eq!(from_str("00"),  Err(SyntaxError(InvalidNumber, 1, 2)));
        assert_eq!(from_str("1."),  Err(SyntaxError(InvalidNumber, 1, 3)));
        assert_eq!(from_str("1e"),  Err(SyntaxError(InvalidNumber, 1, 3)));
        assert_eq!(from_str("1e+"), Err(SyntaxError(InvalidNumber, 1, 4)));

        assert_eq!(from_str("18446744073709551616"), Err(SyntaxError(InvalidNumber, 1, 20)));
        assert_eq!(from_str("-9223372036854775809"), Err(SyntaxError(InvalidNumber, 1, 21)));

        assert_eq!(from_str("3"), Ok(U64(3)));
        assert_eq!(from_str("3.1"), Ok(F64(3.1)));
        assert_eq!(from_str("-1.2"), Ok(F64(-1.2)));
        assert_eq!(from_str("0.4"), Ok(F64(0.4)));
        assert_eq!(from_str("0.4e5"), Ok(F64(0.4e5)));
        assert_eq!(from_str("0.4e+15"), Ok(F64(0.4e15)));
        assert_eq!(from_str("0.4e-01"), Ok(F64(0.4e-01)));
        assert_eq!(from_str(" 3 "), Ok(U64(3)));

        assert_eq!(from_str("-9223372036854775808"), Ok(I64(i64::MIN)));
        assert_eq!(from_str("9223372036854775807"), Ok(U64(i64::MAX as u64)));
        assert_eq!(from_str("18446744073709551615"), Ok(U64(u64::MAX)));
    }

    #[test]
    fn test_decode_numbers() {
        let v: f64 = super::decode("3").unwrap();
        assert_eq!(v, 3.0);

        let v: f64 = super::decode("3.1").unwrap();
        assert_eq!(v, 3.1);

        let v: f64 = super::decode("-1.2").unwrap();
        assert_eq!(v, -1.2);

        let v: f64 = super::decode("0.4").unwrap();
        assert_eq!(v, 0.4);

        let v: f64 = super::decode("0.4e5").unwrap();
        assert_eq!(v, 0.4e5);

        let v: f64 = super::decode("0.4e15").unwrap();
        assert_eq!(v, 0.4e15);

        let v: f64 = super::decode("0.4e-01").unwrap();
        assert_eq!(v, 0.4e-01);

        let v: u64 = super::decode("0").unwrap();
        assert_eq!(v, 0);

        let v: u64 = super::decode("18446744073709551615").unwrap();
        assert_eq!(v, u64::MAX);

        let v: i64 = super::decode("-9223372036854775808").unwrap();
        assert_eq!(v, i64::MIN);

        let v: i64 = super::decode("9223372036854775807").unwrap();
        assert_eq!(v, i64::MAX);

        let res: DecodeResult<i64> = super::decode("765.25");
        assert_eq!(res, Err(ExpectedError("Integer".to_string(),
                                          "765.25".to_string())));
    }

    #[test]
    fn test_read_str() {
        assert_eq!(from_str("\""),    Err(SyntaxError(EOFWhileParsingString, 1, 2)));
        assert_eq!(from_str("\"lol"), Err(SyntaxError(EOFWhileParsingString, 1, 5)));

        assert_eq!(from_str("\"\""), Ok(String("".to_string())));
        assert_eq!(from_str("\"foo\""), Ok(String("foo".to_string())));
        assert_eq!(from_str("\"\\\"\""), Ok(String("\"".to_string())));
        assert_eq!(from_str("\"\\b\""), Ok(String("\x08".to_string())));
        assert_eq!(from_str("\"\\n\""), Ok(String("\n".to_string())));
        assert_eq!(from_str("\"\\r\""), Ok(String("\r".to_string())));
        assert_eq!(from_str("\"\\t\""), Ok(String("\t".to_string())));
        assert_eq!(from_str(" \"foo\" "), Ok(String("foo".to_string())));
        assert_eq!(from_str("\"\\u12ab\""), Ok(String("\u{12ab}".to_string())));
        assert_eq!(from_str("\"\\uAB12\""), Ok(String("\u{AB12}".to_string())));
    }

    #[test]
    fn test_decode_str() {
        let s = [("\"\"", ""),
                 ("\"foo\"", "foo"),
                 ("\"\\\"\"", "\""),
                 ("\"\\b\"", "\x08"),
                 ("\"\\n\"", "\n"),
                 ("\"\\r\"", "\r"),
                 ("\"\\t\"", "\t"),
                 ("\"\\u12ab\"", "\u{12ab}"),
                 ("\"\\uAB12\"", "\u{AB12}")];

        for &(i, o) in &s {
            let v: string::String = super::decode(i).unwrap();
            assert_eq!(v, o);
        }
    }

    #[test]
    fn test_read_array() {
        assert_eq!(from_str("["),     Err(SyntaxError(EOFWhileParsingValue, 1, 2)));
        assert_eq!(from_str("[1"),    Err(SyntaxError(EOFWhileParsingArray, 1, 3)));
        assert_eq!(from_str("[1,"),   Err(SyntaxError(EOFWhileParsingValue, 1, 4)));
        assert_eq!(from_str("[1,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
        assert_eq!(from_str("[6 7]"), Err(SyntaxError(InvalidSyntax,        1, 4)));

        assert_eq!(from_str("[]"), Ok(Array(vec![])));
        assert_eq!(from_str("[ ]"), Ok(Array(vec![])));
        assert_eq!(from_str("[true]"), Ok(Array(vec![Boolean(true)])));
        assert_eq!(from_str("[ false ]"), Ok(Array(vec![Boolean(false)])));
        assert_eq!(from_str("[null]"), Ok(Array(vec![Null])));
        assert_eq!(from_str("[3, 1]"),
                     Ok(Array(vec![U64(3), U64(1)])));
        assert_eq!(from_str("\n[3, 2]\n"),
                     Ok(Array(vec![U64(3), U64(2)])));
        assert_eq!(from_str("[2, [4, 1]]"),
               Ok(Array(vec![U64(2), Array(vec![U64(4), U64(1)])])));
    }

    #[test]
    fn test_decode_array() {
        let v: Vec<()> = super::decode("[]").unwrap();
        assert_eq!(v, []);

        let v: Vec<()> = super::decode("[null]").unwrap();
        assert_eq!(v, [()]);

        let v: Vec<bool> = super::decode("[true]").unwrap();
        assert_eq!(v, [true]);

        let v: Vec<isize> = super::decode("[3, 1]").unwrap();
        assert_eq!(v, [3, 1]);

        let v: Vec<Vec<usize>> = super::decode("[[3], [1, 2]]").unwrap();
        assert_eq!(v, [vec![3], vec![1, 2]]);
    }

    #[test]
    fn test_decode_tuple() {
        let t: (usize, usize, usize) = super::decode("[1, 2, 3]").unwrap();
        assert_eq!(t, (1, 2, 3));

        let t: (usize, string::String) = super::decode("[1, \"two\"]").unwrap();
        assert_eq!(t, (1, "two".to_string()));
    }

    #[test]
    fn test_decode_tuple_malformed_types() {
        assert!(super::decode::<(usize, string::String)>("[1, 2]").is_err());
    }

    #[test]
    fn test_decode_tuple_malformed_length() {
        assert!(super::decode::<(usize, usize)>("[1, 2, 3]").is_err());
    }

    #[test]
    fn test_read_object() {
        assert_eq!(from_str("{"),       Err(SyntaxError(EOFWhileParsingObject, 1, 2)));
        assert_eq!(from_str("{ "),      Err(SyntaxError(EOFWhileParsingObject, 1, 3)));
        assert_eq!(from_str("{1"),      Err(SyntaxError(KeyMustBeAString,      1, 2)));
        assert_eq!(from_str("{ \"a\""), Err(SyntaxError(EOFWhileParsingObject, 1, 6)));
        assert_eq!(from_str("{\"a\""),  Err(SyntaxError(EOFWhileParsingObject, 1, 5)));
        assert_eq!(from_str("{\"a\" "), Err(SyntaxError(EOFWhileParsingObject, 1, 6)));

        assert_eq!(from_str("{\"a\" 1"),   Err(SyntaxError(ExpectedColon,         1, 6)));
        assert_eq!(from_str("{\"a\":"),    Err(SyntaxError(EOFWhileParsingValue,  1, 6)));
        assert_eq!(from_str("{\"a\":1"),   Err(SyntaxError(EOFWhileParsingObject, 1, 7)));
        assert_eq!(from_str("{\"a\":1 1"), Err(SyntaxError(InvalidSyntax,         1, 8)));
        assert_eq!(from_str("{\"a\":1,"),  Err(SyntaxError(EOFWhileParsingObject, 1, 8)));

        assert_eq!(from_str("{}").unwrap(), mk_object(&[]));
        assert_eq!(from_str("{\"a\": 3}").unwrap(),
                  mk_object(&[("a".to_string(), U64(3))]));

        assert_eq!(from_str(
                      "{ \"a\": null, \"b\" : true }").unwrap(),
                  mk_object(&[
                      ("a".to_string(), Null),
                      ("b".to_string(), Boolean(true))]));
        assert_eq!(from_str("\n{ \"a\": null, \"b\" : true }\n").unwrap(),
                  mk_object(&[
                      ("a".to_string(), Null),
                      ("b".to_string(), Boolean(true))]));
        assert_eq!(from_str(
                      "{\"a\" : 1.0 ,\"b\": [ true ]}").unwrap(),
                  mk_object(&[
                      ("a".to_string(), F64(1.0)),
                      ("b".to_string(), Array(vec![Boolean(true)]))
                  ]));
        assert_eq!(from_str(
                      "{\
                          \"a\": 1.0, \
                          \"b\": [\
                              true,\
                              \"foo\\nbar\", \
                              { \"c\": {\"d\": null} } \
                          ]\
                      }").unwrap(),
                  mk_object(&[
                      ("a".to_string(), F64(1.0)),
                      ("b".to_string(), Array(vec![
                          Boolean(true),
                          String("foo\nbar".to_string()),
                          mk_object(&[
                              ("c".to_string(), mk_object(&[("d".to_string(), Null)]))
                          ])
                      ]))
                  ]));
    }

    #[test]
    fn test_decode_struct() {
        let s = "{
            \"inner\": [
                { \"a\": null, \"b\": 2, \"c\": [\"abc\", \"xyz\"] }
            ]
        }";

        let v: Outer = super::decode(s).unwrap();
        assert_eq!(
            v,
            Outer {
                inner: vec![
                    Inner { a: (), b: 2, c: vec!["abc".to_string(), "xyz".to_string()] }
                ]
            }
        );
    }

    #[derive(RustcDecodable)]
    struct FloatStruct {
        f: f64,
        a: Vec<f64>
    }
    #[test]
    fn test_decode_struct_with_nan() {
        let s = "{\"f\":null,\"a\":[null,123]}";
        let obj: FloatStruct = super::decode(s).unwrap();
        assert!(obj.f.is_nan());
        assert!(obj.a[0].is_nan());
        assert_eq!(obj.a[1], 123f64);
    }

    #[test]
    fn test_decode_option() {
        let value: Option<string::String> = super::decode("null").unwrap();
        assert_eq!(value, None);

        let value: Option<string::String> = super::decode("\"jodhpurs\"").unwrap();
        assert_eq!(value, Some("jodhpurs".to_string()));
    }

    #[test]
    fn test_decode_enum() {
        let value: Animal = super::decode("\"Dog\"").unwrap();
        assert_eq!(value, Dog);

        let s = "{\"variant\":\"Frog\",\"fields\":[\"Henry\",349]}";
        let value: Animal = super::decode(s).unwrap();
        assert_eq!(value, Frog("Henry".to_string(), 349));
    }

    #[test]
    fn test_decode_map() {
        let s = "{\"a\": \"Dog\", \"b\": {\"variant\":\"Frog\",\
                  \"fields\":[\"Henry\", 349]}}";
        let mut map: BTreeMap<string::String, Animal> = super::decode(s).unwrap();

        assert_eq!(map.remove(&"a".to_string()), Some(Dog));
        assert_eq!(map.remove(&"b".to_string()), Some(Frog("Henry".to_string(), 349)));
    }

    #[test]
    fn test_multiline_errors() {
        assert_eq!(from_str("{\n  \"foo\":\n \"bar\""),
            Err(SyntaxError(EOFWhileParsingObject, 3, 8)));
    }

    #[derive(RustcDecodable)]
    #[allow(dead_code)]
    struct DecodeStruct {
        x: f64,
        y: bool,
        z: string::String,
        w: Vec<DecodeStruct>
    }
    #[derive(RustcDecodable)]
    enum DecodeEnum {
        A(f64),
        B(string::String)
    }
    fn check_err<T: Decodable>(to_parse: &'static str, expected: DecoderError) {
        let res: DecodeResult<T> = match from_str(to_parse) {
            Err(e) => Err(ParseError(e)),
            Ok(json) => Decodable::decode(&mut Decoder::new(json))
        };
        match res {
            Ok(_) => panic!("`{:?}` parsed & decoded ok, expecting error `{:?}`",
                              to_parse, expected),
            Err(ParseError(e)) => panic!("`{:?}` is not valid json: {:?}",
                                           to_parse, e),
            Err(e) => {
                assert_eq!(e, expected);
            }
        }
    }
    #[test]
    fn test_decode_errors_struct() {
        check_err::<DecodeStruct>("[]", ExpectedError("Object".to_string(), "[]".to_string()));
        check_err::<DecodeStruct>("{\"x\": true, \"y\": true, \"z\": \"\", \"w\": []}",
                                  ExpectedError("Number".to_string(), "true".to_string()));
        check_err::<DecodeStruct>("{\"x\": 1, \"y\": [], \"z\": \"\", \"w\": []}",
                                  ExpectedError("Boolean".to_string(), "[]".to_string()));
        check_err::<DecodeStruct>("{\"x\": 1, \"y\": true, \"z\": {}, \"w\": []}",
                                  ExpectedError("String".to_string(), "{}".to_string()));
        check_err::<DecodeStruct>("{\"x\": 1, \"y\": true, \"z\": \"\", \"w\": null}",
                                  ExpectedError("Array".to_string(), "null".to_string()));
        check_err::<DecodeStruct>("{\"x\": 1, \"y\": true, \"z\": \"\"}",
                                  MissingFieldError("w".to_string()));
    }
    #[test]
    fn test_decode_errors_enum() {
        check_err::<DecodeEnum>("{}",
                                MissingFieldError("variant".to_string()));
        check_err::<DecodeEnum>("{\"variant\": 1}",
                                ExpectedError("String".to_string(), "1".to_string()));
        check_err::<DecodeEnum>("{\"variant\": \"A\"}",
                                MissingFieldError("fields".to_string()));
        check_err::<DecodeEnum>("{\"variant\": \"A\", \"fields\": null}",
                                ExpectedError("Array".to_string(), "null".to_string()));
        check_err::<DecodeEnum>("{\"variant\": \"C\", \"fields\": []}",
                                UnknownVariantError("C".to_string()));
    }

    #[test]
    fn test_find(){
        let json_value = from_str("{\"dog\" : \"cat\"}").unwrap();
        let found_str = json_value.find("dog");
        assert!(found_str.unwrap().as_string().unwrap() == "cat");
    }

    #[test]
    fn test_find_path(){
        let json_value = from_str("{\"dog\":{\"cat\": {\"mouse\" : \"cheese\"}}}").unwrap();
        let found_str = json_value.find_path(&["dog", "cat", "mouse"]);
        assert!(found_str.unwrap().as_string().unwrap() == "cheese");
    }

    #[test]
    fn test_search(){
        let json_value = from_str("{\"dog\":{\"cat\": {\"mouse\" : \"cheese\"}}}").unwrap();
        let found_str = json_value.search("mouse").and_then(|j| j.as_string());
        assert!(found_str.unwrap() == "cheese");
    }

    #[test]
    fn test_index(){
        let json_value = from_str("{\"animals\":[\"dog\",\"cat\",\"mouse\"]}").unwrap();
        let ref array = json_value["animals"];
        assert_eq!(array[0].as_string().unwrap(), "dog");
        assert_eq!(array[1].as_string().unwrap(), "cat");
        assert_eq!(array[2].as_string().unwrap(), "mouse");
    }

    #[test]
    fn test_is_object(){
        let json_value = from_str("{}").unwrap();
        assert!(json_value.is_object());
    }

    #[test]
    fn test_as_object(){
        let json_value = from_str("{}").unwrap();
        let json_object = json_value.as_object();
        assert!(json_object.is_some());
    }

    #[test]
    fn test_is_array(){
        let json_value = from_str("[1, 2, 3]").unwrap();
        assert!(json_value.is_array());
    }

    #[test]
    fn test_as_array(){
        let json_value = from_str("[1, 2, 3]").unwrap();
        let json_array = json_value.as_array();
        let expected_length = 3;
        assert!(json_array.is_some() && json_array.unwrap().len() == expected_length);
    }

    #[test]
    fn test_is_string(){
        let json_value = from_str("\"dog\"").unwrap();
        assert!(json_value.is_string());
    }

    #[test]
    fn test_as_string(){
        let json_value = from_str("\"dog\"").unwrap();
        let json_str = json_value.as_string();
        let expected_str = "dog";
        assert_eq!(json_str, Some(expected_str));
    }

    #[test]
    fn test_is_number(){
        let json_value = from_str("12").unwrap();
        assert!(json_value.is_number());
    }

    #[test]
    fn test_is_i64(){
        let json_value = from_str("-12").unwrap();
        assert!(json_value.is_i64());

        let json_value = from_str("12").unwrap();
        assert!(!json_value.is_i64());

        let json_value = from_str("12.0").unwrap();
        assert!(!json_value.is_i64());
    }

    #[test]
    fn test_is_u64(){
        let json_value = from_str("12").unwrap();
        assert!(json_value.is_u64());

        let json_value = from_str("-12").unwrap();
        assert!(!json_value.is_u64());

        let json_value = from_str("12.0").unwrap();
        assert!(!json_value.is_u64());
    }

    #[test]
    fn test_is_f64(){
        let json_value = from_str("12").unwrap();
        assert!(!json_value.is_f64());

        let json_value = from_str("-12").unwrap();
        assert!(!json_value.is_f64());

        let json_value = from_str("12.0").unwrap();
        assert!(json_value.is_f64());

        let json_value = from_str("-12.0").unwrap();
        assert!(json_value.is_f64());
    }

    #[test]
    fn test_as_i64(){
        let json_value = from_str("-12").unwrap();
        let json_num = json_value.as_i64();
        assert_eq!(json_num, Some(-12));
    }

    #[test]
    fn test_as_u64(){
        let json_value = from_str("12").unwrap();
        let json_num = json_value.as_u64();
        assert_eq!(json_num, Some(12));
    }

    #[test]
    fn test_as_f64(){
        let json_value = from_str("12.0").unwrap();
        let json_num = json_value.as_f64();
        assert_eq!(json_num, Some(12f64));
    }

    #[test]
    fn test_is_boolean(){
        let json_value = from_str("false").unwrap();
        assert!(json_value.is_boolean());
    }

    #[test]
    fn test_as_boolean(){
        let json_value = from_str("false").unwrap();
        let json_bool = json_value.as_boolean();
        let expected_bool = false;
        assert!(json_bool.is_some() && json_bool.unwrap() == expected_bool);
    }

    #[test]
    fn test_is_null(){
        let json_value = from_str("null").unwrap();
        assert!(json_value.is_null());
    }

    #[test]
    fn test_as_null(){
        let json_value = from_str("null").unwrap();
        let json_null = json_value.as_null();
        let expected_null = ();
        assert!(json_null.is_some() && json_null.unwrap() == expected_null);
    }

    #[test]
    fn test_encode_hashmap_with_numeric_key() {
        use std::str::from_utf8;
        use std::collections::HashMap;
        let mut hm: HashMap<usize, bool> = HashMap::new();
        hm.insert(1, true);
        let mut mem_buf = Vec::new();
        write!(&mut mem_buf, "{}", super::as_pretty_json(&hm)).unwrap();
        let json_str = from_utf8(&mem_buf[..]).unwrap();
        match from_str(json_str) {
            Err(_) => panic!("Unable to parse json_str: {:?}", json_str),
            _ => {} // it parsed and we are good to go
        }
    }

    #[test]
    fn test_prettyencode_hashmap_with_numeric_key() {
        use std::str::from_utf8;
        use std::collections::HashMap;
        let mut hm: HashMap<usize, bool> = HashMap::new();
        hm.insert(1, true);
        let mut mem_buf = Vec::new();
        write!(&mut mem_buf, "{}", super::as_pretty_json(&hm)).unwrap();
        let json_str = from_utf8(&mem_buf[..]).unwrap();
        match from_str(json_str) {
            Err(_) => panic!("Unable to parse json_str: {:?}", json_str),
            _ => {} // it parsed and we are good to go
        }
    }

    #[test]
    fn test_prettyencoder_indent_level_param() {
        use std::str::from_utf8;
        use std::collections::BTreeMap;

        let mut tree = BTreeMap::new();

        tree.insert("hello".to_string(), String("guten tag".to_string()));
        tree.insert("goodbye".to_string(), String("sayonara".to_string()));

        let json = Array(
            // The following layout below should look a lot like
            // the pretty-printed JSON (indent * x)
            vec!
            ( // 0x
                String("greetings".to_string()), // 1x
                Object(tree), // 1x + 2x + 2x + 1x
            ) // 0x
            // End JSON array (7 lines)
        );

        // Helper function for counting indents
        fn indents(source: &str) -> usize {
            let trimmed = source.trim_left_matches(' ');
            source.len() - trimmed.len()
        }

        // Test up to 4 spaces of indents (more?)
        for i in 0..4 {
            let mut writer = Vec::new();
            write!(&mut writer, "{}",
                   super::as_pretty_json(&json).indent(i)).unwrap();

            let printed = from_utf8(&writer[..]).unwrap();

            // Check for indents at each line
            let lines: Vec<&str> = printed.lines().collect();
            assert_eq!(lines.len(), 7); // JSON should be 7 lines

            assert_eq!(indents(lines[0]), 0 * i); // [
            assert_eq!(indents(lines[1]), 1 * i); //   "greetings",
            assert_eq!(indents(lines[2]), 1 * i); //   {
            assert_eq!(indents(lines[3]), 2 * i); //     "hello": "guten tag",
            assert_eq!(indents(lines[4]), 2 * i); //     "goodbye": "sayonara"
            assert_eq!(indents(lines[5]), 1 * i); //   },
            assert_eq!(indents(lines[6]), 0 * i); // ]

            // Finally, test that the pretty-printed JSON is valid
            from_str(printed).ok().expect("Pretty-printed JSON is invalid!");
        }
    }

    #[test]
    fn test_hashmap_with_enum_key() {
        use std::collections::HashMap;
        use json;
        #[derive(RustcEncodable, Eq, Hash, PartialEq, RustcDecodable, Debug)]
        enum Enum {
            Foo,
            #[allow(dead_code)]
            Bar,
        }
        let mut map = HashMap::new();
        map.insert(Enum::Foo, 0);
        let result = json::encode(&map).unwrap();
        assert_eq!(&result[..], r#"{"Foo":0}"#);
        let decoded: HashMap<Enum, _> = json::decode(&result).unwrap();
        assert_eq!(map, decoded);
    }

    #[test]
    fn test_hashmap_with_numeric_key_can_handle_double_quote_delimited_key() {
        use std::collections::HashMap;
        use Decodable;
        let json_str = "{\"1\":true}";
        let json_obj = match from_str(json_str) {
            Err(_) => panic!("Unable to parse json_str: {:?}", json_str),
            Ok(o) => o
        };
        let mut decoder = Decoder::new(json_obj);
        let _hm: HashMap<usize, bool> = Decodable::decode(&mut decoder).unwrap();
    }

    #[test]
    fn test_hashmap_with_numeric_key_will_error_with_string_keys() {
        use std::collections::HashMap;
        use Decodable;
        let json_str = "{\"a\":true}";
        let json_obj = match from_str(json_str) {
            Err(_) => panic!("Unable to parse json_str: {:?}", json_str),
            Ok(o) => o
        };
        let mut decoder = Decoder::new(json_obj);
        let result: Result<HashMap<usize, bool>, DecoderError> = Decodable::decode(&mut decoder);
        assert_eq!(result, Err(ExpectedError("Number".to_string(), "a".to_string())));
    }

    fn assert_stream_equal(src: &str,
                           expected: Vec<(JsonEvent, Vec<StackElement>)>) {
        let mut parser = Parser::new(src.chars());
        let mut i = 0;
        loop {
            let evt = match parser.next() {
                Some(e) => e,
                None => { break; }
            };
            let (ref expected_evt, ref expected_stack) = expected[i];
            if !parser.stack().is_equal_to(expected_stack) {
                panic!("Parser stack is not equal to {:?}", expected_stack);
            }
            assert_eq!(&evt, expected_evt);
            i+=1;
        }
    }
    #[test]
    #[cfg_attr(target_pointer_width = "32", ignore)] // FIXME(#14064)
    fn test_streaming_parser() {
        assert_stream_equal(
            r#"{ "foo":"bar", "array" : [0, 1, 2, 3, 4, 5], "idents":[null,true,false]}"#,
            vec![
                (ObjectStart,             vec![]),
                  (StringValue("bar".to_string()),   vec![StackElement::Key("foo")]),
                  (ArrayStart,            vec![StackElement::Key("array")]),
                    (U64Value(0),         vec![StackElement::Key("array"), StackElement::Index(0)]),
                    (U64Value(1),         vec![StackElement::Key("array"), StackElement::Index(1)]),
                    (U64Value(2),         vec![StackElement::Key("array"), StackElement::Index(2)]),
                    (U64Value(3),         vec![StackElement::Key("array"), StackElement::Index(3)]),
                    (U64Value(4),         vec![StackElement::Key("array"), StackElement::Index(4)]),
                    (U64Value(5),         vec![StackElement::Key("array"), StackElement::Index(5)]),
                  (ArrayEnd,              vec![StackElement::Key("array")]),
                  (ArrayStart,            vec![StackElement::Key("idents")]),
                    (NullValue,           vec![StackElement::Key("idents"),
                                               StackElement::Index(0)]),
                    (BooleanValue(true),  vec![StackElement::Key("idents"),
                                               StackElement::Index(1)]),
                    (BooleanValue(false), vec![StackElement::Key("idents"),
                                               StackElement::Index(2)]),
                  (ArrayEnd,              vec![StackElement::Key("idents")]),
                (ObjectEnd,               vec![]),
            ]
        );
    }
    fn last_event(src: &str) -> JsonEvent {
        let mut parser = Parser::new(src.chars());
        let mut evt = NullValue;
        loop {
            evt = match parser.next() {
                Some(e) => e,
                None => return evt,
            }
        }
    }

    #[test]
    #[cfg_attr(target_pointer_width = "32", ignore)] // FIXME(#14064)
    fn test_read_object_streaming() {
        assert_eq!(last_event("{ "),      Error(SyntaxError(EOFWhileParsingObject, 1, 3)));
        assert_eq!(last_event("{1"),      Error(SyntaxError(KeyMustBeAString,      1, 2)));
        assert_eq!(last_event("{ \"a\""), Error(SyntaxError(EOFWhileParsingObject, 1, 6)));
        assert_eq!(last_event("{\"a\""),  Error(SyntaxError(EOFWhileParsingObject, 1, 5)));
        assert_eq!(last_event("{\"a\" "), Error(SyntaxError(EOFWhileParsingObject, 1, 6)));

        assert_eq!(last_event("{\"a\" 1"),   Error(SyntaxError(ExpectedColon,         1, 6)));
        assert_eq!(last_event("{\"a\":"),    Error(SyntaxError(EOFWhileParsingValue,  1, 6)));
        assert_eq!(last_event("{\"a\":1"),   Error(SyntaxError(EOFWhileParsingObject, 1, 7)));
        assert_eq!(last_event("{\"a\":1 1"), Error(SyntaxError(InvalidSyntax,         1, 8)));
        assert_eq!(last_event("{\"a\":1,"),  Error(SyntaxError(EOFWhileParsingObject, 1, 8)));
        assert_eq!(last_event("{\"a\":1,}"), Error(SyntaxError(TrailingComma, 1, 8)));

        assert_stream_equal(
            "{}",
            vec![(ObjectStart, vec![]), (ObjectEnd, vec![])]
        );
        assert_stream_equal(
            "{\"a\": 3}",
            vec![
                (ObjectStart,        vec![]),
                  (U64Value(3),      vec![StackElement::Key("a")]),
                (ObjectEnd,          vec![]),
            ]
        );
        assert_stream_equal(
            "{ \"a\": null, \"b\" : true }",
            vec![
                (ObjectStart,           vec![]),
                  (NullValue,           vec![StackElement::Key("a")]),
                  (BooleanValue(true),  vec![StackElement::Key("b")]),
                (ObjectEnd,             vec![]),
            ]
        );
        assert_stream_equal(
            "{\"a\" : 1.0 ,\"b\": [ true ]}",
            vec![
                (ObjectStart,           vec![]),
                  (F64Value(1.0),       vec![StackElement::Key("a")]),
                  (ArrayStart,          vec![StackElement::Key("b")]),
                    (BooleanValue(true),vec![StackElement::Key("b"), StackElement::Index(0)]),
                  (ArrayEnd,            vec![StackElement::Key("b")]),
                (ObjectEnd,             vec![]),
            ]
        );
        assert_stream_equal(
            r#"{
                "a": 1.0,
                "b": [
                    true,
                    "foo\nbar",
                    { "c": {"d": null} }
                ]
            }"#,
            vec![
                (ObjectStart,                   vec![]),
                  (F64Value(1.0),               vec![StackElement::Key("a")]),
                  (ArrayStart,                  vec![StackElement::Key("b")]),
                    (BooleanValue(true),        vec![StackElement::Key("b"),
                                                     StackElement::Index(0)]),
                    (StringValue("foo\nbar".to_string()),  vec![StackElement::Key("b"),
                                                                StackElement::Index(1)]),
                    (ObjectStart,               vec![StackElement::Key("b"),
                                                     StackElement::Index(2)]),
                      (ObjectStart,             vec![StackElement::Key("b"),
                                                     StackElement::Index(2),
                                                     StackElement::Key("c")]),
                        (NullValue,             vec![StackElement::Key("b"),
                                                     StackElement::Index(2),
                                                     StackElement::Key("c"),
                                                     StackElement::Key("d")]),
                      (ObjectEnd,               vec![StackElement::Key("b"),
                                                     StackElement::Index(2),
                                                     StackElement::Key("c")]),
                    (ObjectEnd,                 vec![StackElement::Key("b"),
                                                     StackElement::Index(2)]),
                  (ArrayEnd,                    vec![StackElement::Key("b")]),
                (ObjectEnd,                     vec![]),
            ]
        );
    }
    #[test]
    #[cfg_attr(target_pointer_width = "32", ignore)] // FIXME(#14064)
    fn test_read_array_streaming() {
        assert_stream_equal(
            "[]",
            vec![
                (ArrayStart, vec![]),
                (ArrayEnd,   vec![]),
            ]
        );
        assert_stream_equal(
            "[ ]",
            vec![
                (ArrayStart, vec![]),
                (ArrayEnd,   vec![]),
            ]
        );
        assert_stream_equal(
            "[true]",
            vec![
                (ArrayStart,             vec![]),
                    (BooleanValue(true), vec![StackElement::Index(0)]),
                (ArrayEnd,               vec![]),
            ]
        );
        assert_stream_equal(
            "[ false ]",
            vec![
                (ArrayStart,              vec![]),
                    (BooleanValue(false), vec![StackElement::Index(0)]),
                (ArrayEnd,                vec![]),
            ]
        );
        assert_stream_equal(
            "[null]",
            vec![
                (ArrayStart,    vec![]),
                    (NullValue, vec![StackElement::Index(0)]),
                (ArrayEnd,      vec![]),
            ]
        );
        assert_stream_equal(
            "[3, 1]",
            vec![
                (ArrayStart,      vec![]),
                    (U64Value(3), vec![StackElement::Index(0)]),
                    (U64Value(1), vec![StackElement::Index(1)]),
                (ArrayEnd,        vec![]),
            ]
        );
        assert_stream_equal(
            "\n[3, 2]\n",
            vec![
                (ArrayStart,      vec![]),
                    (U64Value(3), vec![StackElement::Index(0)]),
                    (U64Value(2), vec![StackElement::Index(1)]),
                (ArrayEnd,        vec![]),
            ]
        );
        assert_stream_equal(
            "[2, [4, 1]]",
            vec![
                (ArrayStart,           vec![]),
                    (U64Value(2),      vec![StackElement::Index(0)]),
                    (ArrayStart,       vec![StackElement::Index(1)]),
                        (U64Value(4),  vec![StackElement::Index(1), StackElement::Index(0)]),
                        (U64Value(1),  vec![StackElement::Index(1), StackElement::Index(1)]),
                    (ArrayEnd,         vec![StackElement::Index(1)]),
                (ArrayEnd,             vec![]),
            ]
        );

        assert_eq!(last_event("["), Error(SyntaxError(EOFWhileParsingValue, 1,  2)));

        assert_eq!(from_str("["),     Err(SyntaxError(EOFWhileParsingValue, 1, 2)));
        assert_eq!(from_str("[1"),    Err(SyntaxError(EOFWhileParsingArray, 1, 3)));
        assert_eq!(from_str("[1,"),   Err(SyntaxError(EOFWhileParsingValue, 1, 4)));
        assert_eq!(from_str("[1,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
        assert_eq!(from_str("[6 7]"), Err(SyntaxError(InvalidSyntax,        1, 4)));

    }
    #[test]
    fn test_trailing_characters_streaming() {
        assert_eq!(last_event("nulla"),  Error(SyntaxError(TrailingCharacters, 1, 5)));
        assert_eq!(last_event("truea"),  Error(SyntaxError(TrailingCharacters, 1, 5)));
        assert_eq!(last_event("falsea"), Error(SyntaxError(TrailingCharacters, 1, 6)));
        assert_eq!(last_event("1a"),     Error(SyntaxError(TrailingCharacters, 1, 2)));
        assert_eq!(last_event("[]a"),    Error(SyntaxError(TrailingCharacters, 1, 3)));
        assert_eq!(last_event("{}a"),    Error(SyntaxError(TrailingCharacters, 1, 3)));
    }
    #[test]
    fn test_read_identifiers_streaming() {
        assert_eq!(Parser::new("null".chars()).next(), Some(NullValue));
        assert_eq!(Parser::new("true".chars()).next(), Some(BooleanValue(true)));
        assert_eq!(Parser::new("false".chars()).next(), Some(BooleanValue(false)));

        assert_eq!(last_event("n"),    Error(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(last_event("nul"),  Error(SyntaxError(InvalidSyntax, 1, 4)));
        assert_eq!(last_event("t"),    Error(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(last_event("truz"), Error(SyntaxError(InvalidSyntax, 1, 4)));
        assert_eq!(last_event("f"),    Error(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(last_event("faz"),  Error(SyntaxError(InvalidSyntax, 1, 3)));
    }

    #[test]
    fn test_stack() {
        let mut stack = Stack::new();

        assert!(stack.is_empty());
        assert!(stack.is_empty());
        assert!(!stack.last_is_index());

        stack.push_index(0);
        stack.bump_index();

        assert!(stack.len() == 1);
        assert!(stack.is_equal_to(&[StackElement::Index(1)]));
        assert!(stack.starts_with(&[StackElement::Index(1)]));
        assert!(stack.ends_with(&[StackElement::Index(1)]));
        assert!(stack.last_is_index());
        assert!(stack.get(0) == StackElement::Index(1));

        stack.push_key("foo".to_string());

        assert!(stack.len() == 2);
        assert!(stack.is_equal_to(&[StackElement::Index(1), StackElement::Key("foo")]));
        assert!(stack.starts_with(&[StackElement::Index(1), StackElement::Key("foo")]));
        assert!(stack.starts_with(&[StackElement::Index(1)]));
        assert!(stack.ends_with(&[StackElement::Index(1), StackElement::Key("foo")]));
        assert!(stack.ends_with(&[StackElement::Key("foo")]));
        assert!(!stack.last_is_index());
        assert!(stack.get(0) == StackElement::Index(1));
        assert!(stack.get(1) == StackElement::Key("foo"));

        stack.push_key("bar".to_string());

        assert!(stack.len() == 3);
        assert!(stack.is_equal_to(&[StackElement::Index(1),
                                    StackElement::Key("foo"),
                                    StackElement::Key("bar")]));
        assert!(stack.starts_with(&[StackElement::Index(1)]));
        assert!(stack.starts_with(&[StackElement::Index(1), StackElement::Key("foo")]));
        assert!(stack.starts_with(&[StackElement::Index(1),
                                    StackElement::Key("foo"),
                                    StackElement::Key("bar")]));
        assert!(stack.ends_with(&[StackElement::Key("bar")]));
        assert!(stack.ends_with(&[StackElement::Key("foo"), StackElement::Key("bar")]));
        assert!(stack.ends_with(&[StackElement::Index(1),
                                  StackElement::Key("foo"),
                                  StackElement::Key("bar")]));
        assert!(!stack.last_is_index());
        assert!(stack.get(0) == StackElement::Index(1));
        assert!(stack.get(1) == StackElement::Key("foo"));
        assert!(stack.get(2) == StackElement::Key("bar"));

        stack.pop();

        assert!(stack.len() == 2);
        assert!(stack.is_equal_to(&[StackElement::Index(1), StackElement::Key("foo")]));
        assert!(stack.starts_with(&[StackElement::Index(1), StackElement::Key("foo")]));
        assert!(stack.starts_with(&[StackElement::Index(1)]));
        assert!(stack.ends_with(&[StackElement::Index(1), StackElement::Key("foo")]));
        assert!(stack.ends_with(&[StackElement::Key("foo")]));
        assert!(!stack.last_is_index());
        assert!(stack.get(0) == StackElement::Index(1));
        assert!(stack.get(1) == StackElement::Key("foo"));
    }

    #[test]
    fn test_to_json() {
        use std::collections::{HashMap,BTreeMap};
        use super::ToJson;

        let array2 = Array(vec!(U64(1), U64(2)));
        let array3 = Array(vec!(U64(1), U64(2), U64(3)));
        let object = {
            let mut tree_map = BTreeMap::new();
            tree_map.insert("a".to_string(), U64(1));
            tree_map.insert("b".to_string(), U64(2));
            Object(tree_map)
        };

        assert_eq!(array2.to_json(), array2);
        assert_eq!(object.to_json(), object);
        assert_eq!(3_isize.to_json(), I64(3));
        assert_eq!(4_i8.to_json(), I64(4));
        assert_eq!(5_i16.to_json(), I64(5));
        assert_eq!(6_i32.to_json(), I64(6));
        assert_eq!(7_i64.to_json(), I64(7));
        assert_eq!(8_usize.to_json(), U64(8));
        assert_eq!(9_u8.to_json(), U64(9));
        assert_eq!(10_u16.to_json(), U64(10));
        assert_eq!(11_u32.to_json(), U64(11));
        assert_eq!(12_u64.to_json(), U64(12));
        assert_eq!(13.0_f32.to_json(), F64(13.0_f64));
        assert_eq!(14.0_f64.to_json(), F64(14.0_f64));
        assert_eq!(().to_json(), Null);
        assert_eq!(f32::INFINITY.to_json(), Null);
        assert_eq!(f64::NAN.to_json(), Null);
        assert_eq!(true.to_json(), Boolean(true));
        assert_eq!(false.to_json(), Boolean(false));
        assert_eq!("abc".to_json(), String("abc".to_string()));
        assert_eq!("abc".to_string().to_json(), String("abc".to_string()));
        assert_eq!((1_usize, 2_usize).to_json(), array2);
        assert_eq!((1_usize, 2_usize, 3_usize).to_json(), array3);
        assert_eq!([1_usize, 2_usize].to_json(), array2);
        assert_eq!((&[1_usize, 2_usize, 3_usize]).to_json(), array3);
        assert_eq!((vec![1_usize, 2_usize]).to_json(), array2);
        assert_eq!(vec!(1_usize, 2_usize, 3_usize).to_json(), array3);
        let mut tree_map = BTreeMap::new();
        tree_map.insert("a".to_string(), 1 as usize);
        tree_map.insert("b".to_string(), 2);
        assert_eq!(tree_map.to_json(), object);
        let mut hash_map = HashMap::new();
        hash_map.insert("a".to_string(), 1 as usize);
        hash_map.insert("b".to_string(), 2);
        assert_eq!(hash_map.to_json(), object);
        assert_eq!(Some(15).to_json(), I64(15));
        assert_eq!(Some(15 as usize).to_json(), U64(15));
        assert_eq!(None::<isize>.to_json(), Null);
    }

    #[test]
    fn test_encode_hashmap_with_arbitrary_key() {
        use std::collections::HashMap;
        #[derive(PartialEq, Eq, Hash, RustcEncodable)]
        struct ArbitraryType(usize);
        let mut hm: HashMap<ArbitraryType, bool> = HashMap::new();
        hm.insert(ArbitraryType(1), true);
        let mut mem_buf = string::String::new();
        let mut encoder = Encoder::new(&mut mem_buf);
        let result = hm.encode(&mut encoder);
        match result.err().unwrap() {
            EncoderError::BadHashmapKey => (),
            _ => panic!("expected bad hash map key")
        }
    }

    #[bench]
    fn bench_streaming_small(b: &mut Bencher) {
        b.iter( || {
            let mut parser = Parser::new(
                r#"{
                    "a": 1.0,
                    "b": [
                        true,
                        "foo\nbar",
                        { "c": {"d": null} }
                    ]
                }"#.chars()
            );
            loop {
                match parser.next() {
                    None => return,
                    _ => {}
                }
            }
        });
    }
    #[bench]
    fn bench_small(b: &mut Bencher) {
        b.iter( || {
            let _ = from_str(r#"{
                "a": 1.0,
                "b": [
                    true,
                    "foo\nbar",
                    { "c": {"d": null} }
                ]
            }"#);
        });
    }

    fn big_json() -> string::String {
        let mut src = "[\n".to_string();
        for _ in 0..500 {
            src.push_str(r#"{ "a": true, "b": null, "c":3.1415, "d": "Hello world", "e": \
                            [1,2,3]},"#);
        }
        src.push_str("{}]");
        return src;
    }

    #[bench]
    fn bench_streaming_large(b: &mut Bencher) {
        let src = big_json();
        b.iter( || {
            let mut parser = Parser::new(src.chars());
            loop {
                match parser.next() {
                    None => return,
                    _ => {}
                }
            }
        });
    }
    #[bench]
    fn bench_large(b: &mut Bencher) {
        let src = big_json();
        b.iter( || { let _ = from_str(&src); });
    }
}
