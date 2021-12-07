// Rust JSON serialization library.
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
//! ```json
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
//! To be able to encode a piece of data, it must implement the `serialize::Encodable` trait.
//! To be able to decode a piece of data, it must implement the `serialize::Decodable` trait.
//! The Rust compiler provides an annotation to automatically generate the code for these traits:
//! `#[derive(Decodable, Encodable)]`
//!
//! The JSON API provides an enum `json::Json` and a trait `ToJson` to encode objects.
//! The `ToJson` trait provides a `to_json` method to convert an object into a `json::Json` value.
//! A `json::Json` value can be encoded as a string or buffer using the functions described above.
//! You can also use the `json::Encoder` object, which implements the `Encoder` trait.
//!
//! When using `ToJson` the `Encodable` trait implementation is not mandatory.
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
//! use rustc_macros::{Decodable, Encodable};
//! use rustc_serialize::json;
//!
//! // Automatically generate `Decodable` and `Encodable` trait implementations
//! #[derive(Decodable, Encodable)]
//! pub struct TestStruct  {
//!     data_int: u8,
//!     data_str: String,
//!     data_vector: Vec<u8>,
//! }
//!
//! let object = TestStruct {
//!     data_int: 1,
//!     data_str: "homura".to_string(),
//!     data_vector: vec![2,3,4,5],
//! };
//!
//! // Serialize using `json::encode`
//! let encoded = json::encode(&object).unwrap();
//!
//! // Deserialize using `json::decode`
//! let decoded: TestStruct = json::decode(&encoded[..]).unwrap();
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
//! use rustc_macros::Encodable;
//! use rustc_serialize::json::{self, ToJson, Json};
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
//! // Only generate `Encodable` trait implementation
//! #[derive(Encodable)]
//! pub struct ComplexNumRecord {
//!     uid: u8,
//!     dsc: String,
//!     val: Json,
//! }
//!
//! let num = ComplexNum { a: 0.0001, b: 12.539 };
//! let data: String = json::encode(&ComplexNumRecord{
//!     uid: 1,
//!     dsc: "test".to_string(),
//!     val: num.to_json(),
//! }).unwrap();
//! println!("data: {}", data);
//! // data: {"uid":1,"dsc":"test","val":"0.0001+12.539i"};
//! ```
//!
//! ### Verbose example of `ToJson` usage
//!
//! ```rust
//! # #![feature(rustc_private)]
//! use rustc_macros::Decodable;
//! use std::collections::BTreeMap;
//! use rustc_serialize::json::{self, Json, ToJson};
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
//! // Serialize using `ToJson`
//! let input_data = TestStruct {
//!     data_int: 1,
//!     data_str: "madoka".to_string(),
//!     data_vector: vec![2,3,4,5],
//! };
//! let json_obj: Json = input_data.to_json();
//! let json_str: String = json_obj.to_string();
//!
//! // Deserialize like before
//! let decoded: TestStruct = json::decode(&json_str).unwrap();
//! ```

use self::DecoderError::*;
use self::ErrorCode::*;
use self::InternalStackElement::*;
use self::JsonEvent::*;
use self::ParserError::*;
use self::ParserState::*;

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::io;
use std::io::prelude::*;
use std::mem::swap;
use std::num::FpCategory as Fp;
use std::ops::Index;
use std::str::FromStr;
use std::string;
use std::{char, fmt, str};

use crate::Encodable;

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

pub struct PrettyJson<'a> {
    inner: &'a Json,
}

pub struct AsJson<'a, T> {
    inner: &'a T,
}
pub struct AsPrettyJson<'a, T> {
    inner: &'a T,
    indent: Option<usize>,
}

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
    ApplicationError(string::String),
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
pub fn decode<T: crate::Decodable<Decoder>>(s: &str) -> DecodeResult<T> {
    let json = match from_str(s) {
        Ok(x) => x,
        Err(e) => return Err(ParseError(e)),
    };

    let mut decoder = Decoder::new(json);
    crate::Decodable::decode(&mut decoder)
}

/// Shortcut function to encode a `T` into a JSON `String`
pub fn encode<T: for<'r> crate::Encodable<Encoder<'r>>>(
    object: &T,
) -> Result<string::String, EncoderError> {
    let mut s = String::new();
    {
        let mut encoder = Encoder::new(&mut s);
        object.encode(&mut encoder)?;
    }
    Ok(s)
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        error_str(*self).fmt(f)
    }
}

fn io_error_to_error(io: io::Error) -> ParserError {
    IoError(io.kind(), io.to_string())
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME this should be a nicer error
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME this should be a nicer error
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for DecoderError {}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME this should be a nicer error
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for EncoderError {}

impl From<fmt::Error> for EncoderError {
    /// Converts a [`fmt::Error`] into `EncoderError`
    ///
    /// This conversion does not allocate memory.
    fn from(err: fmt::Error) -> EncoderError {
        EncoderError::FmtError(err)
    }
}

pub type EncodeResult = Result<(), EncoderError>;
pub type DecodeResult<T> = Result<T, DecoderError>;

fn escape_str(wr: &mut dyn fmt::Write, v: &str) -> EncodeResult {
    wr.write_str("\"")?;

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
            _ => {
                continue;
            }
        };

        if start < i {
            wr.write_str(&v[start..i])?;
        }

        wr.write_str(escaped)?;

        start = i + 1;
    }

    if start != v.len() {
        wr.write_str(&v[start..])?;
    }

    wr.write_str("\"")?;
    Ok(())
}

fn escape_char(writer: &mut dyn fmt::Write, v: char) -> EncodeResult {
    escape_str(writer, v.encode_utf8(&mut [0; 4]))
}

fn spaces(wr: &mut dyn fmt::Write, mut n: usize) -> EncodeResult {
    const BUF: &str = "                ";

    while n >= BUF.len() {
        wr.write_str(BUF)?;
        n -= BUF.len();
    }

    if n > 0 {
        wr.write_str(&BUF[..n])?;
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
    writer: &'a mut (dyn fmt::Write + 'a),
    is_emitting_map_key: bool,
}

impl<'a> Encoder<'a> {
    /// Creates a new JSON encoder whose output will be written to the writer
    /// specified.
    pub fn new(writer: &'a mut dyn fmt::Write) -> Encoder<'a> {
        Encoder { writer, is_emitting_map_key: false }
    }
}

macro_rules! emit_enquoted_if_mapkey {
    ($enc:ident,$e:expr) => {{
        if $enc.is_emitting_map_key {
            write!($enc.writer, "\"{}\"", $e)?;
        } else {
            write!($enc.writer, "{}", $e)?;
        }
        Ok(())
    }};
}

impl<'a> crate::Encoder for Encoder<'a> {
    type Error = EncoderError;

    fn emit_unit(&mut self) -> EncodeResult {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        write!(self.writer, "null")?;
        Ok(())
    }

    fn emit_usize(&mut self, v: usize) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u128(&mut self, v: u128) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u64(&mut self, v: u64) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u32(&mut self, v: u32) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u16(&mut self, v: u16) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u8(&mut self, v: u8) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }

    fn emit_isize(&mut self, v: isize) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i128(&mut self, v: i128) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i64(&mut self, v: i64) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i32(&mut self, v: i32) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i16(&mut self, v: i16) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i8(&mut self, v: i8) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }

    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if v {
            write!(self.writer, "true")?;
        } else {
            write!(self.writer, "false")?;
        }
        Ok(())
    }

    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, fmt_number_or_null(v))
    }
    fn emit_f32(&mut self, v: f32) -> EncodeResult {
        self.emit_f64(f64::from(v))
    }

    fn emit_char(&mut self, v: char) -> EncodeResult {
        escape_char(self.writer, v)
    }
    fn emit_str(&mut self, v: &str) -> EncodeResult {
        escape_str(self.writer, v)
    }
    fn emit_raw_bytes(&mut self, s: &[u8]) -> Result<(), Self::Error> {
        for &c in s.iter() {
            self.emit_u8(c)?;
        }
        Ok(())
    }

    fn emit_enum<F>(&mut self, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        f(self)
    }

    fn emit_enum_variant<F>(&mut self, name: &str, _id: usize, cnt: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        // enums are encoded as strings or objects
        // Bunny => "Bunny"
        // Kangaroo(34,"William") => {"variant": "Kangaroo", "fields": [34,"William"]}
        if cnt == 0 {
            escape_str(self.writer, name)
        } else {
            if self.is_emitting_map_key {
                return Err(EncoderError::BadHashmapKey);
            }
            write!(self.writer, "{{\"variant\":")?;
            escape_str(self.writer, name)?;
            write!(self.writer, ",\"fields\":[")?;
            f(self)?;
            write!(self.writer, "]}}")?;
            Ok(())
        }
    }

    fn emit_fieldless_enum_variant<const ID: usize>(
        &mut self,
        name: &str,
    ) -> Result<(), Self::Error> {
        escape_str(self.writer, name)
    }

    fn emit_enum_variant_arg<F>(&mut self, first: bool, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if !first {
            write!(self.writer, ",")?;
        }
        f(self)
    }

    fn emit_struct<F>(&mut self, _: bool, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        write!(self.writer, "{{")?;
        f(self)?;
        write!(self.writer, "}}")?;
        Ok(())
    }

    fn emit_struct_field<F>(&mut self, name: &str, first: bool, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if !first {
            write!(self.writer, ",")?;
        }
        escape_str(self.writer, name)?;
        write!(self.writer, ":")?;
        f(self)
    }

    fn emit_tuple<F>(&mut self, len: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        self.emit_seq_elt(idx, f)
    }

    fn emit_option<F>(&mut self, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        f(self)
    }
    fn emit_option_none(&mut self) -> EncodeResult {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        self.emit_unit()
    }
    fn emit_option_some<F>(&mut self, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        f(self)
    }

    fn emit_seq<F>(&mut self, _len: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        write!(self.writer, "[")?;
        f(self)?;
        write!(self.writer, "]")?;
        Ok(())
    }

    fn emit_seq_elt<F>(&mut self, idx: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if idx != 0 {
            write!(self.writer, ",")?;
        }
        f(self)
    }

    fn emit_map<F>(&mut self, _len: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        write!(self.writer, "{{")?;
        f(self)?;
        write!(self.writer, "}}")?;
        Ok(())
    }

    fn emit_map_elt_key<F>(&mut self, idx: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if idx != 0 {
            write!(self.writer, ",")?
        }
        self.is_emitting_map_key = true;
        f(self)?;
        self.is_emitting_map_key = false;
        Ok(())
    }

    fn emit_map_elt_val<F>(&mut self, f: F) -> EncodeResult
    where
        F: FnOnce(&mut Encoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        write!(self.writer, ":")?;
        f(self)
    }
}

/// Another encoder for JSON, but prints out human-readable JSON instead of
/// compact data
pub struct PrettyEncoder<'a> {
    writer: &'a mut (dyn fmt::Write + 'a),
    curr_indent: usize,
    indent: usize,
    is_emitting_map_key: bool,
}

impl<'a> PrettyEncoder<'a> {
    /// Creates a new encoder whose output will be written to the specified writer
    pub fn new(writer: &'a mut dyn fmt::Write) -> PrettyEncoder<'a> {
        PrettyEncoder { writer, curr_indent: 0, indent: 2, is_emitting_map_key: false }
    }

    /// Sets the number of spaces to indent for each level.
    /// This is safe to set during encoding.
    pub fn set_indent(&mut self, indent: usize) {
        // self.indent very well could be 0 so we need to use checked division.
        let level = self.curr_indent.checked_div(self.indent).unwrap_or(0);
        self.indent = indent;
        self.curr_indent = level * self.indent;
    }
}

impl<'a> crate::Encoder for PrettyEncoder<'a> {
    type Error = EncoderError;

    fn emit_unit(&mut self) -> EncodeResult {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        write!(self.writer, "null")?;
        Ok(())
    }

    fn emit_usize(&mut self, v: usize) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u128(&mut self, v: u128) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u64(&mut self, v: u64) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u32(&mut self, v: u32) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u16(&mut self, v: u16) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_u8(&mut self, v: u8) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }

    fn emit_isize(&mut self, v: isize) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i128(&mut self, v: i128) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i64(&mut self, v: i64) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i32(&mut self, v: i32) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i16(&mut self, v: i16) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }
    fn emit_i8(&mut self, v: i8) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, v)
    }

    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if v {
            write!(self.writer, "true")?;
        } else {
            write!(self.writer, "false")?;
        }
        Ok(())
    }

    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        emit_enquoted_if_mapkey!(self, fmt_number_or_null(v))
    }
    fn emit_f32(&mut self, v: f32) -> EncodeResult {
        self.emit_f64(f64::from(v))
    }

    fn emit_char(&mut self, v: char) -> EncodeResult {
        escape_char(self.writer, v)
    }
    fn emit_str(&mut self, v: &str) -> EncodeResult {
        escape_str(self.writer, v)
    }
    fn emit_raw_bytes(&mut self, s: &[u8]) -> Result<(), Self::Error> {
        for &c in s.iter() {
            self.emit_u8(c)?;
        }
        Ok(())
    }

    fn emit_enum<F>(&mut self, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        f(self)
    }

    fn emit_enum_variant<F>(&mut self, name: &str, _id: usize, cnt: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if cnt == 0 {
            escape_str(self.writer, name)
        } else {
            if self.is_emitting_map_key {
                return Err(EncoderError::BadHashmapKey);
            }
            writeln!(self.writer, "{{")?;
            self.curr_indent += self.indent;
            spaces(self.writer, self.curr_indent)?;
            write!(self.writer, "\"variant\": ")?;
            escape_str(self.writer, name)?;
            writeln!(self.writer, ",")?;
            spaces(self.writer, self.curr_indent)?;
            writeln!(self.writer, "\"fields\": [")?;
            self.curr_indent += self.indent;
            f(self)?;
            self.curr_indent -= self.indent;
            writeln!(self.writer)?;
            spaces(self.writer, self.curr_indent)?;
            self.curr_indent -= self.indent;
            writeln!(self.writer, "]")?;
            spaces(self.writer, self.curr_indent)?;
            write!(self.writer, "}}")?;
            Ok(())
        }
    }

    fn emit_fieldless_enum_variant<const ID: usize>(
        &mut self,
        name: &str,
    ) -> Result<(), Self::Error> {
        escape_str(self.writer, name)
    }

    fn emit_enum_variant_arg<F>(&mut self, first: bool, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if !first {
            writeln!(self.writer, ",")?;
        }
        spaces(self.writer, self.curr_indent)?;
        f(self)
    }

    fn emit_struct<F>(&mut self, no_fields: bool, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if no_fields {
            write!(self.writer, "{{}}")?;
        } else {
            write!(self.writer, "{{")?;
            self.curr_indent += self.indent;
            f(self)?;
            self.curr_indent -= self.indent;
            writeln!(self.writer)?;
            spaces(self.writer, self.curr_indent)?;
            write!(self.writer, "}}")?;
        }
        Ok(())
    }

    fn emit_struct_field<F>(&mut self, name: &str, first: bool, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if first {
            writeln!(self.writer)?;
        } else {
            writeln!(self.writer, ",")?;
        }
        spaces(self.writer, self.curr_indent)?;
        escape_str(self.writer, name)?;
        write!(self.writer, ": ")?;
        f(self)
    }

    fn emit_tuple<F>(&mut self, len: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        self.emit_seq_elt(idx, f)
    }

    fn emit_option<F>(&mut self, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        f(self)
    }
    fn emit_option_none(&mut self) -> EncodeResult {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        self.emit_unit()
    }
    fn emit_option_some<F>(&mut self, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        f(self)
    }

    fn emit_seq<F>(&mut self, len: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if len == 0 {
            write!(self.writer, "[]")?;
        } else {
            write!(self.writer, "[")?;
            self.curr_indent += self.indent;
            f(self)?;
            self.curr_indent -= self.indent;
            writeln!(self.writer)?;
            spaces(self.writer, self.curr_indent)?;
            write!(self.writer, "]")?;
        }
        Ok(())
    }

    fn emit_seq_elt<F>(&mut self, idx: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if idx == 0 {
            writeln!(self.writer)?;
        } else {
            writeln!(self.writer, ",")?;
        }
        spaces(self.writer, self.curr_indent)?;
        f(self)
    }

    fn emit_map<F>(&mut self, len: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if len == 0 {
            write!(self.writer, "{{}}")?;
        } else {
            write!(self.writer, "{{")?;
            self.curr_indent += self.indent;
            f(self)?;
            self.curr_indent -= self.indent;
            writeln!(self.writer)?;
            spaces(self.writer, self.curr_indent)?;
            write!(self.writer, "}}")?;
        }
        Ok(())
    }

    fn emit_map_elt_key<F>(&mut self, idx: usize, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        if idx == 0 {
            writeln!(self.writer)?;
        } else {
            writeln!(self.writer, ",")?;
        }
        spaces(self.writer, self.curr_indent)?;
        self.is_emitting_map_key = true;
        f(self)?;
        self.is_emitting_map_key = false;
        Ok(())
    }

    fn emit_map_elt_val<F>(&mut self, f: F) -> EncodeResult
    where
        F: FnOnce(&mut PrettyEncoder<'a>) -> EncodeResult,
    {
        if self.is_emitting_map_key {
            return Err(EncoderError::BadHashmapKey);
        }
        write!(self.writer, ": ")?;
        f(self)
    }
}

impl<E: crate::Encoder> Encodable<E> for Json {
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        match *self {
            Json::I64(v) => v.encode(e),
            Json::U64(v) => v.encode(e),
            Json::F64(v) => v.encode(e),
            Json::String(ref v) => v.encode(e),
            Json::Boolean(v) => v.encode(e),
            Json::Array(ref v) => v.encode(e),
            Json::Object(ref v) => v.encode(e),
            Json::Null => e.emit_unit(),
        }
    }
}

/// Creates an `AsJson` wrapper which can be used to print a value as JSON
/// on-the-fly via `write!`
pub fn as_json<T>(t: &T) -> AsJson<'_, T> {
    AsJson { inner: t }
}

/// Creates an `AsPrettyJson` wrapper which can be used to print a value as JSON
/// on-the-fly via `write!`
pub fn as_pretty_json<T>(t: &T) -> AsPrettyJson<'_, T> {
    AsPrettyJson { inner: t, indent: None }
}

impl Json {
    /// Borrow this json object as a pretty object to generate a pretty
    /// representation for it via `Display`.
    pub fn pretty(&self) -> PrettyJson<'_> {
        PrettyJson { inner: self }
    }

    /// If the Json value is an Object, returns the value associated with the provided key.
    /// Otherwise, returns None.
    pub fn find(&self, key: &str) -> Option<&Json> {
        match *self {
            Json::Object(ref map) => map.get(key),
            _ => None,
        }
    }

    /// If the Json value is an Object, deletes the value associated with the
    /// provided key from the Object and returns it. Otherwise, returns None.
    pub fn remove_key(&mut self, key: &str) -> Option<Json> {
        match *self {
            Json::Object(ref mut map) => map.remove(key),
            _ => None,
        }
    }

    /// Attempts to get a nested Json Object for each key in `keys`.
    /// If any key is found not to exist, `find_path` will return `None`.
    /// Otherwise, it will return the Json value associated with the final key.
    pub fn find_path<'a>(&'a self, keys: &[&str]) -> Option<&'a Json> {
        let mut target = self;
        for key in keys {
            target = target.find(*key)?;
        }
        Some(target)
    }

    /// If the Json value is an Object, performs a depth-first search until
    /// a value associated with the provided key is found. If no value is found
    /// or the Json value is not an Object, returns `None`.
    pub fn search(&self, key: &str) -> Option<&Json> {
        match *self {
            Json::Object(ref map) => match map.get(key) {
                Some(json_value) => Some(json_value),
                None => {
                    for v in map.values() {
                        match v.search(key) {
                            x if x.is_some() => return x,
                            _ => (),
                        }
                    }
                    None
                }
            },
            _ => None,
        }
    }

    /// Returns `true` if the Json value is an `Object`.
    pub fn is_object(&self) -> bool {
        self.as_object().is_some()
    }

    /// If the Json value is an `Object`, returns the associated `BTreeMap`;
    /// returns `None` otherwise.
    pub fn as_object(&self) -> Option<&Object> {
        match *self {
            Json::Object(ref map) => Some(map),
            _ => None,
        }
    }

    /// Returns `true` if the Json value is an `Array`.
    pub fn is_array(&self) -> bool {
        self.as_array().is_some()
    }

    /// If the Json value is an `Array`, returns the associated vector;
    /// returns `None` otherwise.
    pub fn as_array(&self) -> Option<&Array> {
        match *self {
            Json::Array(ref array) => Some(&*array),
            _ => None,
        }
    }

    /// Returns `true` if the Json value is a `String`.
    pub fn is_string(&self) -> bool {
        self.as_string().is_some()
    }

    /// If the Json value is a `String`, returns the associated `str`;
    /// returns `None` otherwise.
    pub fn as_string(&self) -> Option<&str> {
        match *self {
            Json::String(ref s) => Some(&s[..]),
            _ => None,
        }
    }

    /// Returns `true` if the Json value is a `Number`.
    pub fn is_number(&self) -> bool {
        matches!(*self, Json::I64(_) | Json::U64(_) | Json::F64(_))
    }

    /// Returns `true` if the Json value is an `i64`.
    pub fn is_i64(&self) -> bool {
        matches!(*self, Json::I64(_))
    }

    /// Returns `true` if the Json value is a `u64`.
    pub fn is_u64(&self) -> bool {
        matches!(*self, Json::U64(_))
    }

    /// Returns `true` if the Json value is a `f64`.
    pub fn is_f64(&self) -> bool {
        matches!(*self, Json::F64(_))
    }

    /// If the Json value is a number, returns or cast it to an `i64`;
    /// returns `None` otherwise.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Json::I64(n) => Some(n),
            Json::U64(n) => Some(n as i64),
            _ => None,
        }
    }

    /// If the Json value is a number, returns or cast it to a `u64`;
    /// returns `None` otherwise.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Json::I64(n) => Some(n as u64),
            Json::U64(n) => Some(n),
            _ => None,
        }
    }

    /// If the Json value is a number, returns or cast it to a `f64`;
    /// returns `None` otherwise.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Json::I64(n) => Some(n as f64),
            Json::U64(n) => Some(n as f64),
            Json::F64(n) => Some(n),
            _ => None,
        }
    }

    /// Returns `true` if the Json value is a `Boolean`.
    pub fn is_boolean(&self) -> bool {
        self.as_boolean().is_some()
    }

    /// If the Json value is a `Boolean`, returns the associated `bool`;
    /// returns `None` otherwise.
    pub fn as_boolean(&self) -> Option<bool> {
        match *self {
            Json::Boolean(b) => Some(b),
            _ => None,
        }
    }

    /// Returns `true` if the Json value is a `Null`.
    pub fn is_null(&self) -> bool {
        self.as_null().is_some()
    }

    /// If the Json value is a `Null`, returns `()`;
    /// returns `None` otherwise.
    pub fn as_null(&self) -> Option<()> {
        match *self {
            Json::Null => Some(()),
            _ => None,
        }
    }
}

impl<'a> Index<&'a str> for Json {
    type Output = Json;

    fn index(&self, idx: &'a str) -> &Json {
        self.find(idx).unwrap()
    }
}

impl Index<usize> for Json {
    type Output = Json;

    fn index(&self, idx: usize) -> &Json {
        match *self {
            Json::Array(ref v) => &v[idx],
            _ => panic!("can only index Json with usize if it is an array"),
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
///
/// An example is `foo.bar[3].x`.
#[derive(Default)]
pub struct Stack {
    stack: Vec<InternalStackElement>,
    str_buffer: Vec<u8>,
}

/// StackElements compose a Stack.
///
/// As an example, `StackElement::Key("foo")`, `StackElement::Key("bar")`,
/// `StackElement::Index(3)`, and `StackElement::Key("x")` are the
/// StackElements composing the stack that represents `foo.bar[3].x`.
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
        Self::default()
    }

    /// Returns The number of elements in the Stack.
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Returns `true` if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Provides access to the StackElement at a given index.
    /// lower indices are at the bottom of the stack while higher indices are
    /// at the top.
    pub fn get(&self, idx: usize) -> StackElement<'_> {
        match self.stack[idx] {
            InternalIndex(i) => StackElement::Index(i),
            InternalKey(start, size) => StackElement::Key(
                str::from_utf8(&self.str_buffer[start as usize..start as usize + size as usize])
                    .unwrap(),
            ),
        }
    }

    /// Compares this stack with an array of StackElement<'_>s.
    pub fn is_equal_to(&self, rhs: &[StackElement<'_>]) -> bool {
        if self.stack.len() != rhs.len() {
            return false;
        }
        for (i, r) in rhs.iter().enumerate() {
            if self.get(i) != *r {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the bottom-most elements of this stack are the same as
    /// the ones passed as parameter.
    pub fn starts_with(&self, rhs: &[StackElement<'_>]) -> bool {
        if self.stack.len() < rhs.len() {
            return false;
        }
        for (i, r) in rhs.iter().enumerate() {
            if self.get(i) != *r {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the top-most elements of this stack are the same as
    /// the ones passed as parameter.
    pub fn ends_with(&self, rhs: &[StackElement<'_>]) -> bool {
        if self.stack.len() < rhs.len() {
            return false;
        }
        let offset = self.stack.len() - rhs.len();
        for (i, r) in rhs.iter().enumerate() {
            if self.get(i + offset) != *r {
                return false;
            }
        }
        true
    }

    /// Returns the top-most element (if any).
    pub fn top(&self) -> Option<StackElement<'_>> {
        match self.stack.last() {
            None => None,
            Some(&InternalIndex(i)) => Some(StackElement::Index(i)),
            Some(&InternalKey(start, size)) => Some(StackElement::Key(
                str::from_utf8(&self.str_buffer[start as usize..(start + size) as usize]).unwrap(),
            )),
        }
    }

    // Used by Parser to insert StackElement::Key elements at the top of the stack.
    fn push_key(&mut self, key: string::String) {
        self.stack.push(InternalKey(self.str_buffer.len() as u16, key.len() as u16));
        self.str_buffer.extend(key.as_bytes());
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
        matches!(self.stack.last(), Some(InternalIndex(_)))
    }

    // Used by Parser to increment the index of the top-most element.
    fn bump_index(&mut self) {
        let len = self.stack.len();
        let idx = match *self.stack.last().unwrap() {
            InternalIndex(i) => i + 1,
            _ => {
                panic!();
            }
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

impl<T: Iterator<Item = char>> Iterator for Parser<T> {
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

impl<T: Iterator<Item = char>> Parser<T> {
    /// Creates the JSON parser.
    pub fn new(rdr: T) -> Parser<T> {
        let mut p = Parser {
            rdr,
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
    pub fn stack(&self) -> &Stack {
        &self.stack
    }

    fn eof(&self) -> bool {
        self.ch.is_none()
    }
    fn ch_or_null(&self) -> char {
        self.ch.unwrap_or('\x00')
    }
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
        while self.ch_is(' ') || self.ch_is('\n') || self.ch_is('\t') || self.ch_is('\r') {
            self.bump();
        }
    }

    fn parse_number(&mut self) -> JsonEvent {
        let neg = if self.ch_is('-') {
            self.bump();
            true
        } else {
            false
        };

        let res = match self.parse_u64() {
            Ok(res) => res,
            Err(e) => {
                return Error(e);
            }
        };

        if self.ch_is('.') || self.ch_is('e') || self.ch_is('E') {
            let mut res = res as f64;

            if self.ch_is('.') {
                res = match self.parse_decimal(res) {
                    Ok(res) => res,
                    Err(e) => {
                        return Error(e);
                    }
                };
            }

            if self.ch_is('e') || self.ch_is('E') {
                res = match self.parse_exponent(res) {
                    Ok(res) => res,
                    Err(e) => {
                        return Error(e);
                    }
                };
            }

            if neg {
                res *= -1.0;
            }

            F64Value(res)
        } else if neg {
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

    fn parse_u64(&mut self) -> Result<u64, ParserError> {
        let mut accum = 0u64;
        let last_accum = 0; // necessary to detect overflow.

        match self.ch_or_null() {
            '0' => {
                self.bump();

                // A leading '0' must be the only digit before the decimal point.
                if let '0'..='9' = self.ch_or_null() {
                    return self.error(InvalidNumber);
                }
            }
            '1'..='9' => {
                while !self.eof() {
                    match self.ch_or_null() {
                        c @ '0'..='9' => {
                            accum = accum.wrapping_mul(10);
                            accum = accum.wrapping_add((c as u64) - ('0' as u64));

                            // Detect overflow by comparing to the last value.
                            if accum <= last_accum {
                                return self.error(InvalidNumber);
                            }

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
            '0'..='9' => (),
            _ => return self.error(InvalidNumber),
        }

        let mut dec = 1.0;
        while !self.eof() {
            match self.ch_or_null() {
                c @ '0'..='9' => {
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
            '0'..='9' => (),
            _ => return self.error(InvalidNumber),
        }
        while !self.eof() {
            match self.ch_or_null() {
                c @ '0'..='9' => {
                    exp *= 10;
                    exp += (c as usize) - ('0' as usize);

                    self.bump();
                }
                _ => break,
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
                c @ '0'..='9' => n * 16 + ((c as u16) - ('0' as u16)),
                'a' | 'A' => n * 16 + 10,
                'b' | 'B' => n * 16 + 11,
                'c' | 'C' => n * 16 + 12,
                'd' | 'D' => n * 16 + 13,
                'e' | 'E' => n * 16 + 14,
                'f' | 'F' => n * 16 + 15,
                _ => return self.error(InvalidEscape),
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
                    'u' => match self.decode_hex_escape()? {
                        0xDC00..=0xDFFF => return self.error(LoneLeadingSurrogateInHexEscape),

                        // Non-BMP characters are encoded as a sequence of
                        // two hex escapes, representing UTF-16 surrogates.
                        n1 @ 0xD800..=0xDBFF => {
                            match (self.next_char(), self.next_char()) {
                                (Some('\\'), Some('u')) => (),
                                _ => return self.error(UnexpectedEndOfHexEscape),
                            }

                            let n2 = self.decode_hex_escape()?;
                            if !(0xDC00..=0xDFFF).contains(&n2) {
                                return self.error(LoneLeadingSurrogateInHexEscape);
                            }
                            let c =
                                (u32::from(n1 - 0xD800) << 10 | u32::from(n2 - 0xDC00)) + 0x1_0000;
                            res.push(char::from_u32(c).unwrap());
                        }

                        n => match char::from_u32(u32::from(n)) {
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
                    }
                    Some(c) => res.push(c),
                    None => unreachable!(),
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
                    if let Some(evt) = self.parse_array_comma_or_end() {
                        return evt;
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
        if self.eof() {
            return self.error_event(EOFWhileParsingValue);
        }
        match self.ch_or_null() {
            'n' => self.parse_ident("ull", NullValue),
            't' => self.parse_ident("rue", BooleanValue(true)),
            'f' => self.parse_ident("alse", BooleanValue(false)),
            '0'..='9' | '-' => self.parse_number(),
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
            _ => self.error_event(InvalidSyntax),
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

impl<T: Iterator<Item = char>> Builder<T> {
    /// Creates a JSON Builder.
    pub fn new(src: T) -> Builder<T> {
        Builder { parser: Parser::new(src), token: None }
    }

    // Decode a Json value from a Parser.
    pub fn build(&mut self) -> Result<Json, BuilderError> {
        self.bump();
        let result = self.build_value();
        self.bump();
        match self.token {
            None => {}
            Some(Error(ref e)) => {
                return Err(e.clone());
            }
            ref tok => {
                panic!("unexpected token {:?}", tok.clone());
            }
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
                Err(e) => return Err(e),
            }
            self.bump();
        }
    }

    fn build_object(&mut self) -> Result<Json, BuilderError> {
        self.bump();

        let mut values = BTreeMap::new();

        loop {
            match self.token {
                Some(ObjectEnd) => {
                    return Ok(Json::Object(values));
                }
                Some(Error(ref e)) => {
                    return Err(e.clone());
                }
                None => {
                    break;
                }
                _ => {}
            }
            let key = match self.parser.stack().top() {
                Some(StackElement::Key(k)) => k.to_owned(),
                _ => {
                    panic!("invalid state");
                }
            };
            match self.build_value() {
                Ok(value) => {
                    values.insert(key, value);
                }
                Err(e) => {
                    return Err(e);
                }
            }
            self.bump();
        }
        self.parser.error(EOFWhileParsingObject)
    }
}

/// Decodes a json value from an `&mut io::Read`
pub fn from_reader(rdr: &mut dyn Read) -> Result<Json, BuilderError> {
    let mut contents = Vec::new();
    match rdr.read_to_end(&mut contents) {
        Ok(c) => c,
        Err(e) => return Err(io_error_to_error(e)),
    };
    let s = match str::from_utf8(&contents).ok() {
        Some(s) => s,
        _ => return Err(SyntaxError(NotUtf8, 0, 0)),
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

    fn pop(&mut self) -> Json {
        self.stack.pop().unwrap()
    }
}

macro_rules! expect {
    ($e:expr, Null) => {{
        match $e {
            Json::Null => Ok(()),
            other => Err(ExpectedError("Null".to_owned(), other.to_string())),
        }
    }};
    ($e:expr, $t:ident) => {{
        match $e {
            Json::$t(v) => Ok(v),
            other => Err(ExpectedError(stringify!($t).to_owned(), other.to_string())),
        }
    }};
}

macro_rules! read_primitive {
    ($name:ident, $ty:ty) => {
        fn $name(&mut self) -> DecodeResult<$ty> {
            match self.pop() {
                Json::I64(f) => Ok(f as $ty),
                Json::U64(f) => Ok(f as $ty),
                Json::F64(f) => Err(ExpectedError("Integer".to_owned(), f.to_string())),
                // re: #12967.. a type w/ numeric keys (ie HashMap<usize, V> etc)
                // is going to have a string here, as per JSON spec.
                Json::String(s) => match s.parse().ok() {
                    Some(f) => Ok(f),
                    None => Err(ExpectedError("Number".to_owned(), s)),
                },
                value => Err(ExpectedError("Number".to_owned(), value.to_string())),
            }
        }
    };
}

impl crate::Decoder for Decoder {
    type Error = DecoderError;

    fn read_nil(&mut self) -> DecodeResult<()> {
        expect!(self.pop(), Null)
    }

    read_primitive! { read_usize, usize }
    read_primitive! { read_u8, u8 }
    read_primitive! { read_u16, u16 }
    read_primitive! { read_u32, u32 }
    read_primitive! { read_u64, u64 }
    read_primitive! { read_u128, u128 }
    read_primitive! { read_isize, isize }
    read_primitive! { read_i8, i8 }
    read_primitive! { read_i16, i16 }
    read_primitive! { read_i32, i32 }
    read_primitive! { read_i64, i64 }
    read_primitive! { read_i128, i128 }

    fn read_f32(&mut self) -> DecodeResult<f32> {
        self.read_f64().map(|x| x as f32)
    }

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
            }
            Json::Null => Ok(f64::NAN),
            value => Err(ExpectedError("Number".to_owned(), value.to_string())),
        }
    }

    fn read_bool(&mut self) -> DecodeResult<bool> {
        expect!(self.pop(), Boolean)
    }

    fn read_char(&mut self) -> DecodeResult<char> {
        let s = self.read_str()?;
        {
            let mut it = s.chars();
            if let (Some(c), None) = (it.next(), it.next()) {
                // exactly one character
                return Ok(c);
            }
        }
        Err(ExpectedError("single character string".to_owned(), s.to_string()))
    }

    fn read_str(&mut self) -> DecodeResult<Cow<'_, str>> {
        expect!(self.pop(), String).map(Cow::Owned)
    }

    fn read_raw_bytes_into(&mut self, s: &mut [u8]) -> Result<(), Self::Error> {
        for c in s.iter_mut() {
            *c = self.read_u8()?;
        }
        Ok(())
    }

    fn read_enum<T, F>(&mut self, f: F) -> DecodeResult<T>
    where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        f(self)
    }

    fn read_enum_variant<T, F>(&mut self, names: &[&str], mut f: F) -> DecodeResult<T>
    where
        F: FnMut(&mut Decoder, usize) -> DecodeResult<T>,
    {
        let name = match self.pop() {
            Json::String(s) => s,
            Json::Object(mut o) => {
                let n = match o.remove("variant") {
                    Some(Json::String(s)) => s,
                    Some(val) => return Err(ExpectedError("String".to_owned(), val.to_string())),
                    None => return Err(MissingFieldError("variant".to_owned())),
                };
                match o.remove("fields") {
                    Some(Json::Array(l)) => {
                        self.stack.extend(l.into_iter().rev());
                    }
                    Some(val) => return Err(ExpectedError("Array".to_owned(), val.to_string())),
                    None => return Err(MissingFieldError("fields".to_owned())),
                }
                n
            }
            json => return Err(ExpectedError("String or Object".to_owned(), json.to_string())),
        };
        let idx = match names.iter().position(|n| *n == &name[..]) {
            Some(idx) => idx,
            None => return Err(UnknownVariantError(name)),
        };
        f(self, idx)
    }

    fn read_enum_variant_arg<T, F>(&mut self, f: F) -> DecodeResult<T>
    where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        f(self)
    }

    fn read_struct<T, F>(&mut self, f: F) -> DecodeResult<T>
    where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        let value = f(self)?;
        self.pop();
        Ok(value)
    }

    fn read_struct_field<T, F>(&mut self, name: &str, f: F) -> DecodeResult<T>
    where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        let mut obj = expect!(self.pop(), Object)?;

        let value = match obj.remove(name) {
            None => {
                // Add a Null and try to parse it as an Option<_>
                // to get None as a default value.
                self.stack.push(Json::Null);
                match f(self) {
                    Ok(x) => x,
                    Err(_) => return Err(MissingFieldError(name.to_string())),
                }
            }
            Some(json) => {
                self.stack.push(json);
                f(self)?
            }
        };
        self.stack.push(Json::Object(obj));
        Ok(value)
    }

    fn read_tuple<T, F>(&mut self, tuple_len: usize, f: F) -> DecodeResult<T>
    where
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

    fn read_tuple_arg<T, F>(&mut self, f: F) -> DecodeResult<T>
    where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        self.read_seq_elt(f)
    }

    fn read_option<T, F>(&mut self, mut f: F) -> DecodeResult<T>
    where
        F: FnMut(&mut Decoder, bool) -> DecodeResult<T>,
    {
        match self.pop() {
            Json::Null => f(self, false),
            value => {
                self.stack.push(value);
                f(self, true)
            }
        }
    }

    fn read_seq<T, F>(&mut self, f: F) -> DecodeResult<T>
    where
        F: FnOnce(&mut Decoder, usize) -> DecodeResult<T>,
    {
        let array = expect!(self.pop(), Array)?;
        let len = array.len();
        self.stack.extend(array.into_iter().rev());
        f(self, len)
    }

    fn read_seq_elt<T, F>(&mut self, f: F) -> DecodeResult<T>
    where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        f(self)
    }

    fn read_map<T, F>(&mut self, f: F) -> DecodeResult<T>
    where
        F: FnOnce(&mut Decoder, usize) -> DecodeResult<T>,
    {
        let obj = expect!(self.pop(), Object)?;
        let len = obj.len();
        for (key, value) in obj {
            self.stack.push(value);
            self.stack.push(Json::String(key));
        }
        f(self, len)
    }

    fn read_map_elt_key<T, F>(&mut self, f: F) -> DecodeResult<T>
    where
        F: FnOnce(&mut Decoder) -> DecodeResult<T>,
    {
        f(self)
    }

    fn read_map_elt_val<T, F>(&mut self, f: F) -> DecodeResult<T>
    where
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
    fn to_json(&self) -> Json {
        self.clone()
    }
}

impl ToJson for f32 {
    fn to_json(&self) -> Json {
        f64::from(*self).to_json()
    }
}

impl ToJson for f64 {
    fn to_json(&self) -> Json {
        match self.classify() {
            Fp::Nan | Fp::Infinite => Json::Null,
            _ => Json::F64(*self),
        }
    }
}

impl ToJson for () {
    fn to_json(&self) -> Json {
        Json::Null
    }
}

impl ToJson for bool {
    fn to_json(&self) -> Json {
        Json::Boolean(*self)
    }
}

impl ToJson for str {
    fn to_json(&self) -> Json {
        Json::String(self.to_string())
    }
}

impl ToJson for string::String {
    fn to_json(&self) -> Json {
        Json::String((*self).clone())
    }
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

tuple_impl! {A}
tuple_impl! {A, B}
tuple_impl! {A, B, C}
tuple_impl! {A, B, C, D}
tuple_impl! {A, B, C, D, E}
tuple_impl! {A, B, C, D, E, F}
tuple_impl! {A, B, C, D, E, F, G}
tuple_impl! {A, B, C, D, E, F, G, H}
tuple_impl! {A, B, C, D, E, F, G, H, I}
tuple_impl! {A, B, C, D, E, F, G, H, I, J}
tuple_impl! {A, B, C, D, E, F, G, H, I, J, K}
tuple_impl! {A, B, C, D, E, F, G, H, I, J, K, L}

impl<A: ToJson> ToJson for [A] {
    fn to_json(&self) -> Json {
        Json::Array(self.iter().map(|elt| elt.to_json()).collect())
    }
}

impl<A: ToJson> ToJson for Vec<A> {
    fn to_json(&self) -> Json {
        Json::Array(self.iter().map(|elt| elt.to_json()).collect())
    }
}

impl<T: ToString, A: ToJson> ToJson for BTreeMap<T, A> {
    fn to_json(&self) -> Json {
        let mut d = BTreeMap::new();
        for (key, value) in self {
            d.insert(key.to_string(), value.to_json());
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

impl<A: ToJson> ToJson for Option<A> {
    fn to_json(&self) -> Json {
        match *self {
            None => Json::Null,
            Some(ref value) => value.to_json(),
        }
    }
}

struct FormatShim<'a, 'b> {
    inner: &'a mut fmt::Formatter<'b>,
}

impl<'a, 'b> fmt::Write for FormatShim<'a, 'b> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        match self.inner.write_str(s) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error),
        }
    }
}

impl fmt::Display for Json {
    /// Encodes a json value into a string
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut shim = FormatShim { inner: f };
        let mut encoder = Encoder::new(&mut shim);
        match self.encode(&mut encoder) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error),
        }
    }
}

impl<'a> fmt::Display for PrettyJson<'a> {
    /// Encodes a json value into a string
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut shim = FormatShim { inner: f };
        let mut encoder = PrettyEncoder::new(&mut shim);
        match self.inner.encode(&mut encoder) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error),
        }
    }
}

impl<'a, T: for<'r> Encodable<Encoder<'r>>> fmt::Display for AsJson<'a, T> {
    /// Encodes a json value into a string
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut shim = FormatShim { inner: f };
        let mut encoder = Encoder::new(&mut shim);
        match self.inner.encode(&mut encoder) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error),
        }
    }
}

impl<'a, T> AsPrettyJson<'a, T> {
    /// Sets the indentation level for the emitted JSON
    pub fn indent(mut self, indent: usize) -> AsPrettyJson<'a, T> {
        self.indent = Some(indent);
        self
    }
}

impl<'a, T: for<'x> Encodable<PrettyEncoder<'x>>> fmt::Display for AsPrettyJson<'a, T> {
    /// Encodes a json value into a string
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut shim = FormatShim { inner: f };
        let mut encoder = PrettyEncoder::new(&mut shim);
        if let Some(n) = self.indent {
            encoder.set_indent(n);
        }
        match self.inner.encode(&mut encoder) {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error),
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
mod tests;
