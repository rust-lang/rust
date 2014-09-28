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
#![allow(missing_doc)]

/*!
JSON parsing and serialization

# What is JSON?

JSON (JavaScript Object Notation) is a way to write data in Javascript.
Like XML, it allows to encode structured data in a text format that can be easily read by humans.
Its simple syntax and native compatibility with JavaScript have made it a widely used format.

Data types that can be encoded are JavaScript types (see the `Json` enum for more details):

* `Boolean`: equivalent to rust's `bool`
* `Number`: equivalent to rust's `f64`
* `String`: equivalent to rust's `String`
* `List`: equivalent to rust's `Vec<T>`, but also allowing objects of different types in the same
array
* `Object`: equivalent to rust's `Treemap<String, json::Json>`
* `Null`

An object is a series of string keys mapping to values, in `"key": value` format.
Arrays are enclosed in square brackets ([ ... ]) and objects in curly brackets ({ ... }).
A simple JSON document encoding a person, his/her age, address and phone numbers could look like:

```ignore
{
    "FirstName": "John",
    "LastName": "Doe",
    "Age": 43,
    "Address": {
        "Street": "Downing Street 10",
        "City": "London",
        "Country": "Great Britain"
    },
    "PhoneNumbers": [
        "+44 1234567",
        "+44 2345678"
    ]
}
```

# Rust Type-based Encoding and Decoding

Rust provides a mechanism for low boilerplate encoding & decoding of values to and from JSON via
the serialization API.
To be able to encode a piece of data, it must implement the `serialize::Encodable` trait.
To be able to decode a piece of data, it must implement the `serialize::Decodable` trait.
The Rust compiler provides an annotation to automatically generate the code for these traits:
`#[deriving(Decodable, Encodable)]`

The JSON API provides an enum `json::Json` and a trait `ToJson` to encode objects.
The `ToJson` trait provides a `to_json` method to convert an object into a `json::Json` value.
A `json::Json` value can be encoded as a string or buffer using the functions described above.
You can also use the `json::Encoder` object, which implements the `Encoder` trait.

When using `ToJson` the `Encodable` trait implementation is not mandatory.

# Examples of use

## Using Autoserialization

Create a struct called `TestStruct` and serialize and deserialize it to and from JSON using the
serialization API, using the derived serialization code.

```rust
extern crate serialize;
use serialize::json;

// Automatically generate `Decodable` and `Encodable` trait implementations
#[deriving(Decodable, Encodable)]
pub struct TestStruct  {
    data_int: u8,
    data_str: String,
    data_vector: Vec<u8>,
}

fn main() {
    let object = TestStruct {
        data_int: 1,
        data_str: "toto".to_string(),
        data_vector: vec![2,3,4,5],
    };

    // Serialize using `json::encode`
    let encoded = json::encode(&object);

    // Deserialize using `json::decode`
    let decoded: TestStruct = json::decode(encoded.as_slice()).unwrap();
}
```

## Using the `ToJson` trait

The examples above use the `ToJson` trait to generate the JSON string, which is required
for custom mappings.

### Simple example of `ToJson` usage

```rust
extern crate serialize;
use serialize::json::ToJson;
use serialize::json;

// A custom data structure
struct ComplexNum {
    a: f64,
    b: f64,
}

// JSON value representation
impl ToJson for ComplexNum {
    fn to_json(&self) -> json::Json {
        json::String(format!("{}+{}i", self.a, self.b))
    }
}

// Only generate `Encodable` trait implementation
#[deriving(Encodable)]
pub struct ComplexNumRecord {
    uid: u8,
    dsc: String,
    val: json::Json,
}

fn main() {
    let num = ComplexNum { a: 0.0001, b: 12.539 };
    let data: String = json::encode(&ComplexNumRecord{
        uid: 1,
        dsc: "test".to_string(),
        val: num.to_json(),
    });
    println!("data: {}", data);
    // data: {"uid":1,"dsc":"test","val":"0.0001+12.539j"};
}
```

### Verbose example of `ToJson` usage

```rust
extern crate serialize;
use std::collections::TreeMap;
use serialize::json::ToJson;
use serialize::json;

// Only generate `Decodable` trait implementation
#[deriving(Decodable)]
pub struct TestStruct {
    data_int: u8,
    data_str: String,
    data_vector: Vec<u8>,
}

// Specify encoding method manually
impl ToJson for TestStruct {
    fn to_json(&self) -> json::Json {
        let mut d = TreeMap::new();
        // All standard types implement `to_json()`, so use it
        d.insert("data_int".to_string(), self.data_int.to_json());
        d.insert("data_str".to_string(), self.data_str.to_json());
        d.insert("data_vector".to_string(), self.data_vector.to_json());
        json::Object(d)
    }
}

fn main() {
    // Serialize using `ToJson`
    let input_data = TestStruct {
        data_int: 1,
        data_str: "toto".to_string(),
        data_vector: vec![2,3,4,5],
    };
    let json_obj: json::Json = input_data.to_json();
    let json_str: String = json_obj.to_string();

    // Deserialize like before
    let decoded: TestStruct = json::decode(json_str.as_slice()).unwrap();
}
```

*/

use std;
use std::collections::{HashMap, TreeMap};
use std::{char, f64, fmt, io, num, str};
use std::io::MemWriter;
use std::mem::{swap, transmute};
use std::num::{FPNaN, FPInfinite};
use std::str::ScalarValue;
use std::string;
use std::vec::Vec;

use Encodable;

/// Represents a json value
#[deriving(Clone, PartialEq, PartialOrd)]
pub enum Json {
    I64(i64),
    U64(u64),
    F64(f64),
    String(string::String),
    Boolean(bool),
    List(JsonList),
    Object(JsonObject),
    Null,
}

pub type JsonList = Vec<Json>;
pub type JsonObject = TreeMap<string::String, Json>;

/// The errors that can arise while parsing a JSON stream.
#[deriving(Clone, PartialEq)]
pub enum ErrorCode {
    InvalidSyntax,
    InvalidNumber,
    EOFWhileParsingObject,
    EOFWhileParsingList,
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

#[deriving(Clone, PartialEq, Show)]
pub enum ParserError {
    /// msg, line, col
    SyntaxError(ErrorCode, uint, uint),
    IoError(io::IoErrorKind, &'static str),
}

// Builder and Parser have the same errors.
pub type BuilderError = ParserError;

#[deriving(Clone, PartialEq, Show)]
pub enum DecoderError {
    ParseError(ParserError),
    ExpectedError(string::String, string::String),
    MissingFieldError(string::String),
    UnknownVariantError(string::String),
    ApplicationError(string::String)
}

/// Returns a readable error string for a given error code.
pub fn error_str(error: ErrorCode) -> &'static str {
    return match error {
        InvalidSyntax => "invalid syntax",
        InvalidNumber => "invalid number",
        EOFWhileParsingObject => "EOF While parsing object",
        EOFWhileParsingList => "EOF While parsing list",
        EOFWhileParsingValue => "EOF While parsing value",
        EOFWhileParsingString => "EOF While parsing string",
        KeyMustBeAString => "key must be a string",
        ExpectedColon => "expected `:`",
        TrailingCharacters => "trailing characters",
        TrailingComma => "trailing comma",
        InvalidEscape => "invalid escape",
        UnrecognizedHex => "invalid \\u escape (unrecognized hex)",
        NotFourDigit => "invalid \\u escape (not four digits)",
        NotUtf8 => "contents not utf-8",
        InvalidUnicodeCodePoint => "invalid Unicode code point",
        LoneLeadingSurrogateInHexEscape => "lone leading surrogate in hex escape",
        UnexpectedEndOfHexEscape => "unexpected end of hex escape",
    }
}

/// Shortcut function to decode a JSON `&str` into an object
pub fn decode<T: ::Decodable<Decoder, DecoderError>>(s: &str) -> DecodeResult<T> {
    let json = match from_str(s) {
        Ok(x) => x,
        Err(e) => return Err(ParseError(e))
    };

    let mut decoder = Decoder::new(json);
    ::Decodable::decode(&mut decoder)
}

/// Shortcut function to encode a `T` into a JSON `String`
pub fn encode<'a, T: Encodable<Encoder<'a>, io::IoError>>(object: &T) -> string::String {
    let buff = Encoder::buffer_encode(object);
    string::String::from_utf8(buff).unwrap()
}

impl fmt::Show for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        error_str(*self).fmt(f)
    }
}

fn io_error_to_error(io: io::IoError) -> ParserError {
    IoError(io.kind, io.desc)
}

pub type EncodeResult = io::IoResult<()>;
pub type DecodeResult<T> = Result<T, DecoderError>;

pub fn escape_bytes(wr: &mut io::Writer, bytes: &[u8]) -> Result<(), io::IoError> {
    try!(wr.write_str("\""));

    let mut start = 0;

    for (i, byte) in bytes.iter().enumerate() {
        let escaped = match *byte {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            b'\x08' => "\\b",
            b'\x0c' => "\\f",
            b'\n' => "\\n",
            b'\r' => "\\r",
            b'\t' => "\\t",
            _ => { continue; }
        };

        if start < i {
            try!(wr.write(bytes.slice(start, i)));
        }

        try!(wr.write_str(escaped));

        start = i + 1;
    }

    if start != bytes.len() {
        try!(wr.write(bytes.slice_from(start)));
    }

    wr.write_str("\"")
}

fn escape_str(writer: &mut io::Writer, v: &str) -> Result<(), io::IoError> {
    escape_bytes(writer, v.as_bytes())
}

fn escape_char(writer: &mut io::Writer, v: char) -> Result<(), io::IoError> {
    let mut buf = [0, .. 4];
    v.encode_utf8(buf);
    escape_bytes(writer, buf)
}

fn spaces(wr: &mut io::Writer, mut n: uint) -> Result<(), io::IoError> {
    static len: uint = 16;
    static buf: [u8, ..len] = [b' ', ..len];

    while n >= len {
        try!(wr.write(buf));
        n -= len;
    }

    if n > 0 {
        wr.write(buf.slice_to(n))
    } else {
        Ok(())
    }
}

fn fmt_number_or_null(v: f64) -> string::String {
    match v.classify() {
        FPNaN | FPInfinite => string::String::from_str("null"),
        _ => f64::to_str_digits(v, 6u)
    }
}

/// A structure for implementing serialization to JSON.
pub struct Encoder<'a> {
    writer: &'a mut io::Writer+'a,
}

impl<'a> Encoder<'a> {
    /// Creates a new JSON encoder whose output will be written to the writer
    /// specified.
    pub fn new(writer: &'a mut io::Writer) -> Encoder<'a> {
        Encoder { writer: writer }
    }

    /// Encode the specified struct into a json [u8]
    pub fn buffer_encode<T:Encodable<Encoder<'a>, io::IoError>>(object: &T) -> Vec<u8>  {
        //Serialize the object in a string using a writer
        let mut m = MemWriter::new();
        // FIXME(14302) remove the transmute and unsafe block.
        unsafe {
            let mut encoder = Encoder::new(&mut m as &mut io::Writer);
            // MemWriter never Errs
            let _ = object.encode(transmute(&mut encoder));
        }
        m.unwrap()
    }

    /// Encode the specified struct into a json str
    ///
    /// Note: this function is deprecated. Consider using `json::encode` instead.
    #[deprecated = "Replaced by `json::encode`"]
    pub fn str_encode<T: Encodable<Encoder<'a>, io::IoError>>(object: &T) -> string::String {
        encode(object)
    }
}

impl<'a> ::Encoder<io::IoError> for Encoder<'a> {
    fn emit_nil(&mut self) -> EncodeResult { write!(self.writer, "null") }

    fn emit_uint(&mut self, v: uint) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_u64(&mut self, v: u64) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_u32(&mut self, v: u32) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_u16(&mut self, v: u16) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_u8(&mut self, v: u8) -> EncodeResult  { self.emit_f64(v as f64) }

    fn emit_int(&mut self, v: int) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_i64(&mut self, v: i64) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_i32(&mut self, v: i32) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_i16(&mut self, v: i16) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_i8(&mut self, v: i8) -> EncodeResult  { self.emit_f64(v as f64) }

    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        if v {
            write!(self.writer, "true")
        } else {
            write!(self.writer, "false")
        }
    }

    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        write!(self.writer, "{}", fmt_number_or_null(v))
    }
    fn emit_f32(&mut self, v: f32) -> EncodeResult { self.emit_f64(v as f64) }

    fn emit_char(&mut self, v: char) -> EncodeResult {
        escape_char(self.writer, v)
    }
    fn emit_str(&mut self, v: &str) -> EncodeResult {
        escape_str(self.writer, v)
    }

    fn emit_enum(&mut self, _name: &str, f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        f(self)
    }

    fn emit_enum_variant(&mut self,
                         name: &str,
                         _id: uint,
                         cnt: uint,
                         f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        // enums are encoded as strings or objects
        // Bunny => "Bunny"
        // Kangaroo(34,"William") => {"variant": "Kangaroo", "fields": [34,"William"]}
        if cnt == 0 {
            escape_str(self.writer, name)
        } else {
            try!(write!(self.writer, "{{\"variant\":"));
            try!(escape_str(self.writer, name));
            try!(write!(self.writer, ",\"fields\":["));
            try!(f(self));
            write!(self.writer, "]}}")
        }
    }

    fn emit_enum_variant_arg(&mut self,
                             idx: uint,
                             f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        if idx != 0 {
            try!(write!(self.writer, ","));
        }
        f(self)
    }

    fn emit_enum_struct_variant(&mut self,
                                name: &str,
                                id: uint,
                                cnt: uint,
                                f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_enum_variant(name, id, cnt, f)
    }

    fn emit_enum_struct_variant_field(&mut self,
                                      _: &str,
                                      idx: uint,
                                      f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_enum_variant_arg(idx, f)
    }

    fn emit_struct(&mut self,
                   _: &str,
                   _: uint,
                   f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        try!(write!(self.writer, "{{"));
        try!(f(self));
        write!(self.writer, "}}")
    }

    fn emit_struct_field(&mut self,
                         name: &str,
                         idx: uint,
                         f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        if idx != 0 { try!(write!(self.writer, ",")); }
        try!(escape_str(self.writer, name));
        try!(write!(self.writer, ":"));
        f(self)
    }

    fn emit_tuple(&mut self, len: uint, f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg(&mut self,
                      idx: uint,
                      f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_seq_elt(idx, f)
    }

    fn emit_tuple_struct(&mut self,
                         _name: &str,
                         len: uint,
                         f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_seq(len, f)
    }
    fn emit_tuple_struct_arg(&mut self,
                             idx: uint,
                             f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_seq_elt(idx, f)
    }

    fn emit_option(&mut self, f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        f(self)
    }
    fn emit_option_none(&mut self) -> EncodeResult { self.emit_nil() }
    fn emit_option_some(&mut self, f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        f(self)
    }

    fn emit_seq(&mut self, _len: uint, f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        try!(write!(self.writer, "["));
        try!(f(self));
        write!(self.writer, "]")
    }

    fn emit_seq_elt(&mut self, idx: uint, f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        if idx != 0 {
            try!(write!(self.writer, ","));
        }
        f(self)
    }

    fn emit_map(&mut self, _len: uint, f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        try!(write!(self.writer, "{{"));
        try!(f(self));
        write!(self.writer, "}}")
    }

    fn emit_map_elt_key(&mut self,
                        idx: uint,
                        f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        if idx != 0 { try!(write!(self.writer, ",")) }
        // ref #12967, make sure to wrap a key in double quotes,
        // in the event that its of a type that omits them (eg numbers)
        let mut buf = MemWriter::new();
        // FIXME(14302) remove the transmute and unsafe block.
        unsafe {
            let mut check_encoder = Encoder::new(&mut buf);
            try!(f(transmute(&mut check_encoder)));
        }
        let out = str::from_utf8(buf.get_ref()).unwrap();
        let needs_wrapping = out.char_at(0) != '"' && out.char_at_reverse(out.len()) != '"';
        if needs_wrapping { try!(write!(self.writer, "\"")); }
        try!(f(self));
        if needs_wrapping { try!(write!(self.writer, "\"")); }
        Ok(())
    }

    fn emit_map_elt_val(&mut self,
                        _idx: uint,
                        f: |&mut Encoder<'a>| -> EncodeResult) -> EncodeResult {
        try!(write!(self.writer, ":"));
        f(self)
    }
}

/// Another encoder for JSON, but prints out human-readable JSON instead of
/// compact data
pub struct PrettyEncoder<'a> {
    writer: &'a mut io::Writer+'a,
    curr_indent: uint,
    indent: uint,
}

impl<'a> PrettyEncoder<'a> {
    /// Creates a new encoder whose output will be written to the specified writer
    pub fn new<'a>(writer: &'a mut io::Writer) -> PrettyEncoder<'a> {
        PrettyEncoder { writer: writer, curr_indent: 0, indent: 2, }
    }

    /// Set the number of spaces to indent for each level.
    /// This is safe to set during encoding.
    pub fn set_indent<'a>(&mut self, indent: uint) {
        // self.indent very well could be 0 so we need to use checked division.
        let level = self.curr_indent.checked_div(&self.indent).unwrap_or(0);
        self.indent = indent;
        self.curr_indent = level * self.indent;
    }
}

impl<'a> ::Encoder<io::IoError> for PrettyEncoder<'a> {
    fn emit_nil(&mut self) -> EncodeResult { write!(self.writer, "null") }

    fn emit_uint(&mut self, v: uint) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_u64(&mut self, v: u64) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_u32(&mut self, v: u32) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_u16(&mut self, v: u16) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_u8(&mut self, v: u8) -> EncodeResult { self.emit_f64(v as f64) }

    fn emit_int(&mut self, v: int) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_i64(&mut self, v: i64) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_i32(&mut self, v: i32) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_i16(&mut self, v: i16) -> EncodeResult { self.emit_f64(v as f64) }
    fn emit_i8(&mut self, v: i8) -> EncodeResult { self.emit_f64(v as f64) }

    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        if v {
            write!(self.writer, "true")
        } else {
            write!(self.writer, "false")
        }
    }

    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        write!(self.writer, "{}", fmt_number_or_null(v))
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

    fn emit_enum(&mut self,
                 _name: &str,
                 f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        f(self)
    }

    fn emit_enum_variant(&mut self,
                         name: &str,
                         _: uint,
                         cnt: uint,
                         f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        if cnt == 0 {
            escape_str(self.writer, name)
        } else {
            self.curr_indent += self.indent;
            try!(write!(self.writer, "[\n"));
            try!(spaces(self.writer, self.curr_indent));
            try!(escape_str(self.writer, name));
            try!(write!(self.writer, ",\n"));
            try!(f(self));
            self.curr_indent -= self.indent;
            try!(write!(self.writer, "\n"));
            try!(spaces(self.writer, self.curr_indent));
            write!(self.writer, "]")
        }
    }

    fn emit_enum_variant_arg(&mut self,
                             idx: uint,
                             f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        if idx != 0 {
            try!(write!(self.writer, ",\n"));
        }
        try!(spaces(self.writer, self.curr_indent));
        f(self)
    }

    fn emit_enum_struct_variant(&mut self,
                                name: &str,
                                id: uint,
                                cnt: uint,
                                f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_enum_variant(name, id, cnt, f)
    }

    fn emit_enum_struct_variant_field(&mut self,
                                      _: &str,
                                      idx: uint,
                                      f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_enum_variant_arg(idx, f)
    }


    fn emit_struct(&mut self,
                   _: &str,
                   len: uint,
                   f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        if len == 0 {
            write!(self.writer, "{{}}")
        } else {
            try!(write!(self.writer, "{{"));
            self.curr_indent += self.indent;
            try!(f(self));
            self.curr_indent -= self.indent;
            try!(write!(self.writer, "\n"));
            try!(spaces(self.writer, self.curr_indent));
            write!(self.writer, "}}")
        }
    }

    fn emit_struct_field(&mut self,
                         name: &str,
                         idx: uint,
                         f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
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

    fn emit_tuple(&mut self,
                  len: uint,
                  f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg(&mut self,
                      idx: uint,
                      f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_seq_elt(idx, f)
    }

    fn emit_tuple_struct(&mut self,
                         _: &str,
                         len: uint,
                         f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_seq(len, f)
    }
    fn emit_tuple_struct_arg(&mut self,
                             idx: uint,
                             f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        self.emit_seq_elt(idx, f)
    }

    fn emit_option(&mut self, f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        f(self)
    }
    fn emit_option_none(&mut self) -> EncodeResult { self.emit_nil() }
    fn emit_option_some(&mut self, f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        f(self)
    }

    fn emit_seq(&mut self,
                len: uint,
                f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        if len == 0 {
            write!(self.writer, "[]")
        } else {
            try!(write!(self.writer, "["));
            self.curr_indent += self.indent;
            try!(f(self));
            self.curr_indent -= self.indent;
            try!(write!(self.writer, "\n"));
            try!(spaces(self.writer, self.curr_indent));
            write!(self.writer, "]")
        }
    }

    fn emit_seq_elt(&mut self,
                    idx: uint,
                    f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        if idx == 0 {
            try!(write!(self.writer, "\n"));
        } else {
            try!(write!(self.writer, ",\n"));
        }
        try!(spaces(self.writer, self.curr_indent));
        f(self)
    }

    fn emit_map(&mut self,
                len: uint,
                f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        if len == 0 {
            write!(self.writer, "{{}}")
        } else {
            try!(write!(self.writer, "{{"));
            self.curr_indent += self.indent;
            try!(f(self));
            self.curr_indent -= self.indent;
            try!(write!(self.writer, "\n"));
            try!(spaces(self.writer, self.curr_indent));
            write!(self.writer, "}}")
        }
    }

    fn emit_map_elt_key(&mut self,
                        idx: uint,
                        f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        if idx == 0 {
            try!(write!(self.writer, "\n"));
        } else {
            try!(write!(self.writer, ",\n"));
        }
        try!(spaces(self.writer, self.curr_indent));
        // ref #12967, make sure to wrap a key in double quotes,
        // in the event that its of a type that omits them (eg numbers)
        let mut buf = MemWriter::new();
        // FIXME(14302) remove the transmute and unsafe block.
        unsafe {
            let mut check_encoder = PrettyEncoder::new(&mut buf);
            try!(f(transmute(&mut check_encoder)));
        }
        let out = str::from_utf8(buf.get_ref()).unwrap();
        let needs_wrapping = out.char_at(0) != '"' && out.char_at_reverse(out.len()) != '"';
        if needs_wrapping { try!(write!(self.writer, "\"")); }
        try!(f(self));
        if needs_wrapping { try!(write!(self.writer, "\"")); }
        Ok(())
    }

    fn emit_map_elt_val(&mut self,
                        _idx: uint,
                        f: |&mut PrettyEncoder<'a>| -> EncodeResult) -> EncodeResult {
        try!(write!(self.writer, ": "));
        f(self)
    }
}

impl<E: ::Encoder<S>, S> Encodable<E, S> for Json {
    fn encode(&self, e: &mut E) -> Result<(), S> {
        match *self {
            I64(v) => v.encode(e),
            U64(v) => v.encode(e),
            F64(v) => v.encode(e),
            String(ref v) => v.encode(e),
            Boolean(v) => v.encode(e),
            List(ref v) => v.encode(e),
            Object(ref v) => v.encode(e),
            Null => e.emit_nil(),
        }
    }
}

impl Json {
    /// Encodes a json value into an io::writer. Uses a single line.
    pub fn to_writer(&self, writer: &mut io::Writer) -> EncodeResult {
        let mut encoder = Encoder::new(writer);
        self.encode(&mut encoder)
    }

    /// Encodes a json value into an io::writer.
    /// Pretty-prints in a more readable format.
    pub fn to_pretty_writer(&self, writer: &mut io::Writer) -> EncodeResult {
        let mut encoder = PrettyEncoder::new(writer);
        self.encode(&mut encoder)
    }

    /// Encodes a json value into a string
    pub fn to_pretty_str(&self) -> string::String {
        let mut s = MemWriter::new();
        self.to_pretty_writer(&mut s as &mut io::Writer).unwrap();
        string::String::from_utf8(s.unwrap()).unwrap()
    }

     /// If the Json value is an Object, returns the value associated with the provided key.
    /// Otherwise, returns None.
    pub fn find<'a>(&'a self, key: &string::String) -> Option<&'a Json>{
        match self {
            &Object(ref map) => map.find(key),
            _ => None
        }
    }

    /// Attempts to get a nested Json Object for each key in `keys`.
    /// If any key is found not to exist, find_path will return None.
    /// Otherwise, it will return the Json value associated with the final key.
    pub fn find_path<'a>(&'a self, keys: &[&string::String]) -> Option<&'a Json>{
        let mut target = self;
        for key in keys.iter() {
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
    pub fn search<'a>(&'a self, key: &string::String) -> Option<&'a Json> {
        match self {
            &Object(ref map) => {
                match map.find(key) {
                    Some(json_value) => Some(json_value),
                    None => {
                        let mut value : Option<&'a Json> = None;
                        for (_, v) in map.iter() {
                            value = v.search(key);
                            if value.is_some() {
                                break;
                            }
                        }
                        value
                    }
                }
            },
            _ => None
        }
    }

    /// Returns true if the Json value is an Object. Returns false otherwise.
    pub fn is_object<'a>(&'a self) -> bool {
        self.as_object().is_some()
    }

    /// If the Json value is an Object, returns the associated TreeMap.
    /// Returns None otherwise.
    pub fn as_object<'a>(&'a self) -> Option<&'a JsonObject> {
        match self {
            &Object(ref map) => Some(map),
            _ => None
        }
    }

    /// Returns true if the Json value is a List. Returns false otherwise.
    pub fn is_list<'a>(&'a self) -> bool {
        self.as_list().is_some()
    }

    /// If the Json value is a List, returns the associated vector.
    /// Returns None otherwise.
    pub fn as_list<'a>(&'a self) -> Option<&'a JsonList> {
        match self {
            &List(ref list) => Some(&*list),
            _ => None
        }
    }

    /// Returns true if the Json value is a String. Returns false otherwise.
    pub fn is_string<'a>(&'a self) -> bool {
        self.as_string().is_some()
    }

    /// If the Json value is a String, returns the associated str.
    /// Returns None otherwise.
    pub fn as_string<'a>(&'a self) -> Option<&'a str> {
        match *self {
            String(ref s) => Some(s.as_slice()),
            _ => None
        }
    }

    /// Returns true if the Json value is a Number. Returns false otherwise.
    pub fn is_number(&self) -> bool {
        match *self {
            I64(_) | U64(_) | F64(_) => true,
            _ => false,
        }
    }

    /// Returns true if the Json value is a i64. Returns false otherwise.
    pub fn is_i64(&self) -> bool {
        match *self {
            I64(_) => true,
            _ => false,
        }
    }

    /// Returns true if the Json value is a u64. Returns false otherwise.
    pub fn is_u64(&self) -> bool {
        match *self {
            U64(_) => true,
            _ => false,
        }
    }

    /// Returns true if the Json value is a f64. Returns false otherwise.
    pub fn is_f64(&self) -> bool {
        match *self {
            F64(_) => true,
            _ => false,
        }
    }

    /// If the Json value is a number, return or cast it to a i64.
    /// Returns None otherwise.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            I64(n) => Some(n),
            U64(n) => num::cast(n),
            _ => None
        }
    }

    /// If the Json value is a number, return or cast it to a u64.
    /// Returns None otherwise.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            I64(n) => num::cast(n),
            U64(n) => Some(n),
            _ => None
        }
    }

    /// If the Json value is a number, return or cast it to a f64.
    /// Returns None otherwise.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            I64(n) => num::cast(n),
            U64(n) => num::cast(n),
            F64(n) => Some(n),
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
        match self {
            &Boolean(b) => Some(b),
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
        match self {
            &Null => Some(()),
            _ => None
        }
    }
}

/// The output of the streaming parser.
#[deriving(PartialEq, Clone, Show)]
pub enum JsonEvent {
    ObjectStart,
    ObjectEnd,
    ListStart,
    ListEnd,
    BooleanValue(bool),
    I64Value(i64),
    U64Value(u64),
    F64Value(f64),
    StringValue(string::String),
    NullValue,
    Error(ParserError),
}

#[deriving(PartialEq, Show)]
enum ParserState {
    // Parse a value in a list, true means first element.
    ParseArray(bool),
    // Parse ',' or ']' after an element in a list.
    ParseListComma,
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
/// For example, Key("foo"), Key("bar"), Index(3) and Key("x") are the
/// StackElements compositing the stack that represents foo.bar[3].x
#[deriving(PartialEq, Clone, Show)]
pub enum StackElement<'l> {
    Index(u32),
    Key(&'l str),
}

// Internally, Key elements are stored as indices in a buffer to avoid
// allocating a string for every member of an object.
#[deriving(PartialEq, Clone, Show)]
enum InternalStackElement {
    InternalIndex(u32),
    InternalKey(u16, u16), // start, size
}

impl Stack {
    pub fn new() -> Stack {
        Stack { stack: Vec::new(), str_buffer: Vec::new() }
    }

    /// Returns The number of elements in the Stack.
    pub fn len(&self) -> uint { self.stack.len() }

    /// Returns true if the stack is empty.
    pub fn is_empty(&self) -> bool { self.stack.is_empty() }

    /// Provides access to the StackElement at a given index.
    /// lower indices are at the bottom of the stack while higher indices are
    /// at the top.
    pub fn get<'l>(&'l self, idx: uint) -> StackElement<'l> {
        match self.stack[idx] {
            InternalIndex(i) => { Index(i) }
            InternalKey(start, size) => {
                Key(str::from_utf8(
                    self.str_buffer.slice(start as uint, start as uint + size as uint)).unwrap())
            }
        }
    }

    /// Compares this stack with an array of StackElements.
    pub fn is_equal_to(&self, rhs: &[StackElement]) -> bool {
        if self.stack.len() != rhs.len() { return false; }
        for i in range(0, rhs.len()) {
            if self.get(i) != rhs[i] { return false; }
        }
        return true;
    }

    /// Returns true if the bottom-most elements of this stack are the same as
    /// the ones passed as parameter.
    pub fn starts_with(&self, rhs: &[StackElement]) -> bool {
        if self.stack.len() < rhs.len() { return false; }
        for i in range(0, rhs.len()) {
            if self.get(i) != rhs[i] { return false; }
        }
        return true;
    }

    /// Returns true if the top-most elements of this stack are the same as
    /// the ones passed as parameter.
    pub fn ends_with(&self, rhs: &[StackElement]) -> bool {
        if self.stack.len() < rhs.len() { return false; }
        let offset = self.stack.len() - rhs.len();
        for i in range(0, rhs.len()) {
            if self.get(i + offset) != rhs[i] { return false; }
        }
        return true;
    }

    /// Returns the top-most element (if any).
    pub fn top<'l>(&'l self) -> Option<StackElement<'l>> {
        return match self.stack.last() {
            None => None,
            Some(&InternalIndex(i)) => Some(Index(i)),
            Some(&InternalKey(start, size)) => {
                Some(Key(str::from_utf8(
                    self.str_buffer.slice(start as uint, (start+size) as uint)
                ).unwrap()))
            }
        }
    }

    // Used by Parser to insert Key elements at the top of the stack.
    fn push_key(&mut self, key: string::String) {
        self.stack.push(InternalKey(self.str_buffer.len() as u16, key.len() as u16));
        for c in key.as_bytes().iter() {
            self.str_buffer.push(*c);
        }
    }

    // Used by Parser to insert Index elements at the top of the stack.
    fn push_index(&mut self, index: u32) {
        self.stack.push(InternalIndex(index));
    }

    // Used by Parser to remove the top-most element of the stack.
    fn pop(&mut self) {
        assert!(!self.is_empty());
        match *self.stack.last().unwrap() {
            InternalKey(_, sz) => {
                let new_size = self.str_buffer.len() - sz as uint;
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
            _ => { fail!(); }
        };
        *self.stack.get_mut(len - 1) = InternalIndex(idx);
    }
}

/// A streaming JSON parser implemented as an iterator of JsonEvent, consuming
/// an iterator of char.
pub struct Parser<T> {
    rdr: T,
    ch: Option<char>,
    line: uint,
    col: uint,
    // We maintain a stack representing where we are in the logical structure
    // of the JSON stream.
    stack: Stack,
    // A state machine is kept to make it possible to interrupt and resume parsing.
    state: ParserState,
}

impl<T: Iterator<char>> Iterator<JsonEvent> for Parser<T> {
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

        return Some(self.parse());
    }
}

impl<T: Iterator<char>> Parser<T> {
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
        return p;
    }

    /// Provides access to the current position in the logical structure of the
    /// JSON stream.
    pub fn stack<'l>(&'l self) -> &'l Stack {
        return &self.stack;
    }

    fn eof(&self) -> bool { self.ch.is_none() }
    fn ch_or_null(&self) -> char { self.ch.unwrap_or('\x00') }
    fn bump(&mut self) {
        self.ch = self.rdr.next();

        if self.ch_is('\n') {
            self.line += 1u;
            self.col = 1u;
        } else {
            self.col += 1u;
        }
    }

    fn next_char(&mut self) -> Option<char> {
        self.bump();
        self.ch
    }
    fn ch_is(&self, c: char) -> bool {
        self.ch == Some(c)
    }

    fn error<T>(&self, reason: ErrorCode) -> Result<T, ParserError> {
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
                let res = -(res as i64);

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
        let mut accum = 0;
        let last_accum = 0; // necessary to detect overflow.

        match self.ch_or_null() {
            '0' => {
                self.bump();

                // A leading '0' must be the only digit before the decimal point.
                match self.ch_or_null() {
                    '0' .. '9' => return self.error(InvalidNumber),
                    _ => ()
                }
            },
            '1' .. '9' => {
                while !self.eof() {
                    match self.ch_or_null() {
                        c @ '0' .. '9' => {
                            accum *= 10;
                            accum += (c as u64) - ('0' as u64);

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
            '0' .. '9' => (),
             _ => return self.error(InvalidNumber)
        }

        let mut dec = 1.0;
        while !self.eof() {
            match self.ch_or_null() {
                c @ '0' .. '9' => {
                    dec /= 10.0;
                    res += (((c as int) - ('0' as int)) as f64) * dec;
                    self.bump();
                }
                _ => break,
            }
        }

        Ok(res)
    }

    fn parse_exponent(&mut self, mut res: f64) -> Result<f64, ParserError> {
        self.bump();

        let mut exp = 0u;
        let mut neg_exp = false;

        if self.ch_is('+') {
            self.bump();
        } else if self.ch_is('-') {
            self.bump();
            neg_exp = true;
        }

        // Make sure a digit follows the exponent place.
        match self.ch_or_null() {
            '0' .. '9' => (),
            _ => return self.error(InvalidNumber)
        }
        while !self.eof() {
            match self.ch_or_null() {
                c @ '0' .. '9' => {
                    exp *= 10;
                    exp += (c as uint) - ('0' as uint);

                    self.bump();
                }
                _ => break
            }
        }

        let exp = num::pow(10_f64, exp);
        if neg_exp {
            res /= exp;
        } else {
            res *= exp;
        }

        Ok(res)
    }

    fn decode_hex_escape(&mut self) -> Result<u16, ParserError> {
        let mut i = 0u;
        let mut n = 0u16;
        while i < 4 && !self.eof() {
            self.bump();
            n = match self.ch_or_null() {
                c @ '0' .. '9' => n * 16 + ((c as u16) - ('0' as u16)),
                'a' | 'A' => n * 16 + 10,
                'b' | 'B' => n * 16 + 11,
                'c' | 'C' => n * 16 + 12,
                'd' | 'D' => n * 16 + 13,
                'e' | 'E' => n * 16 + 14,
                'f' | 'F' => n * 16 + 15,
                _ => return self.error(InvalidEscape)
            };

            i += 1u;
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
                        0xDC00 .. 0xDFFF => return self.error(LoneLeadingSurrogateInHexEscape),

                        // Non-BMP characters are encoded as a sequence of
                        // two hex escapes, representing UTF-16 surrogates.
                        n1 @ 0xD800 .. 0xDBFF => {
                            match (self.next_char(), self.next_char()) {
                                (Some('\\'), Some('u')) => (),
                                _ => return self.error(UnexpectedEndOfHexEscape),
                            }

                            let buf = [n1, try!(self.decode_hex_escape())];
                            match str::utf16_items(buf.as_slice()).next() {
                                Some(ScalarValue(c)) => res.push(c),
                                _ => return self.error(LoneLeadingSurrogateInHexEscape),
                            }
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
    // stream int the form of a stack that can be queried by the user using the
    // stack() method.
    fn parse(&mut self) -> JsonEvent {
        loop {
            // The only paths where the loop can spin a new iteration
            // are in the cases ParseListComma and ParseObjectComma if ','
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
                    return self.parse_list(first);
                }
                ParseListComma => {
                    match self.parse_list_comma_or_end() {
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
            Error(_) => { ParseFinished }
            ListStart => { ParseArray(true) }
            ObjectStart => { ParseObject(true) }
            _ => { ParseBeforeFinish }
        };
        return val;
    }

    fn parse_list(&mut self, first: bool) -> JsonEvent {
        if self.ch_is(']') {
            if !first {
                return self.error_event(InvalidSyntax);
            }
            if self.stack.is_empty() {
                self.state = ParseBeforeFinish;
            } else {
                self.state = if self.stack.last_is_index() {
                    ParseListComma
                } else {
                    ParseObjectComma
                }
            }
            self.bump();
            return ListEnd;
        }
        if first {
            self.stack.push_index(0);
        }

        let val = self.parse_value();

        self.state = match val {
            Error(_) => { ParseFinished }
            ListStart => { ParseArray(true) }
            ObjectStart => { ParseObject(true) }
            _ => { ParseListComma }
        };
        return val;
    }

    fn parse_list_comma_or_end(&mut self) -> Option<JsonEvent> {
        if self.ch_is(',') {
            self.stack.bump_index();
            self.state = ParseArray(false);
            self.bump();
            return None;
        } else if self.ch_is(']') {
            self.stack.pop();
            if self.stack.is_empty() {
                self.state = ParseBeforeFinish;
            } else {
                self.state = if self.stack.last_is_index() {
                    ParseListComma
                } else {
                    ParseObjectComma
                }
            }
            self.bump();
            return Some(ListEnd);
        } else if self.eof() {
            return Some(self.error_event(EOFWhileParsingList));
        } else {
            return Some(self.error_event(InvalidSyntax));
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
            if self.stack.is_empty() {
                self.state = ParseBeforeFinish;
            } else {
                self.state = if self.stack.last_is_index() {
                    ParseListComma
                } else {
                    ParseObjectComma
                }
            }
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
            Ok(s) => { s }
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
            Error(_) => { ParseFinished }
            ListStart => { ParseArray(true) }
            ObjectStart => { ParseObject(true) }
            _ => { ParseObjectComma }
        };
        return val;
    }

    fn parse_object_end(&mut self) -> JsonEvent {
        if self.ch_is('}') {
            if self.stack.is_empty() {
                self.state = ParseBeforeFinish;
            } else {
                self.state = if self.stack.last_is_index() {
                    ParseListComma
                } else {
                    ParseObjectComma
                }
            }
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
            '0' .. '9' | '-' => self.parse_number(),
            '"' => match self.parse_str() {
                Ok(s) => StringValue(s),
                Err(e) => Error(e),
            },
            '[' => {
                self.bump();
                ListStart
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

impl<T: Iterator<char>> Builder<T> {
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
            Some(Error(e)) => { return Err(e); }
            ref tok => { fail!("unexpected token {}", tok.clone()); }
        }
        result
    }

    fn bump(&mut self) {
        self.token = self.parser.next();
    }

    fn build_value(&mut self) -> Result<Json, BuilderError> {
        return match self.token {
            Some(NullValue) => { Ok(Null) }
            Some(I64Value(n)) => { Ok(I64(n)) }
            Some(U64Value(n)) => { Ok(U64(n)) }
            Some(F64Value(n)) => { Ok(F64(n)) }
            Some(BooleanValue(b)) => { Ok(Boolean(b)) }
            Some(StringValue(ref mut s)) => {
                let mut temp = string::String::new();
                swap(s, &mut temp);
                Ok(String(temp))
            }
            Some(Error(e)) => { Err(e) }
            Some(ListStart) => { self.build_list() }
            Some(ObjectStart) => { self.build_object() }
            Some(ObjectEnd) => { self.parser.error(InvalidSyntax) }
            Some(ListEnd) => { self.parser.error(InvalidSyntax) }
            None => { self.parser.error(EOFWhileParsingValue) }
        }
    }

    fn build_list(&mut self) -> Result<Json, BuilderError> {
        self.bump();
        let mut values = Vec::new();

        loop {
            if self.token == Some(ListEnd) {
                return Ok(List(values.into_iter().collect()));
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

        let mut values = TreeMap::new();

        loop {
            match self.token {
                Some(ObjectEnd) => { return Ok(Object(values)); }
                Some(Error(e)) => { return Err(e); }
                None => { break; }
                _ => {}
            }
            let key = match self.parser.stack().top() {
                Some(Key(k)) => { k.to_string() }
                _ => { fail!("invalid state"); }
            };
            match self.build_value() {
                Ok(value) => { values.insert(key, value); }
                Err(e) => { return Err(e); }
            }
            self.bump();
        }
        return self.parser.error(EOFWhileParsingObject);
    }
}

/// Decodes a json value from an `&mut io::Reader`
pub fn from_reader(rdr: &mut io::Reader) -> Result<Json, BuilderError> {
    let contents = match rdr.read_to_end() {
        Ok(c)  => c,
        Err(e) => return Err(io_error_to_error(e))
    };
    let s = match str::from_utf8(contents.as_slice()) {
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

macro_rules! expect(
    ($e:expr, Null) => ({
        match $e {
            Null => Ok(()),
            other => Err(ExpectedError("Null".to_string(),
                                       format!("{}", other)))
        }
    });
    ($e:expr, $t:ident) => ({
        match $e {
            $t(v) => Ok(v),
            other => {
                Err(ExpectedError(stringify!($t).to_string(),
                                  format!("{}", other)))
            }
        }
    })
)

macro_rules! read_primitive {
    ($name:ident, $ty:ty) => {
        fn $name(&mut self) -> DecodeResult<$ty> {
            match self.pop() {
                I64(f) => {
                    match num::cast(f) {
                        Some(f) => Ok(f),
                        None => Err(ExpectedError("Number".to_string(), format!("{}", f))),
                    }
                }
                U64(f) => {
                    match num::cast(f) {
                        Some(f) => Ok(f),
                        None => Err(ExpectedError("Number".to_string(), format!("{}", f))),
                    }
                }
                F64(f) => {
                    match num::cast(f) {
                        Some(f) => Ok(f),
                        None => Err(ExpectedError("Number".to_string(), format!("{}", f))),
                    }
                }
                String(s) => {
                    // re: #12967.. a type w/ numeric keys (ie HashMap<uint, V> etc)
                    // is going to have a string here, as per JSON spec.
                    match std::from_str::from_str(s.as_slice()) {
                        Some(f) => Ok(f),
                        None => Err(ExpectedError("Number".to_string(), s)),
                    }
                },
                value => Err(ExpectedError("Number".to_string(), format!("{}", value)))
            }
        }
    }
}

impl ::Decoder<DecoderError> for Decoder {
    fn read_nil(&mut self) -> DecodeResult<()> {
        debug!("read_nil");
        expect!(self.pop(), Null)
    }

    read_primitive!(read_uint, uint)
    read_primitive!(read_u8, u8)
    read_primitive!(read_u16, u16)
    read_primitive!(read_u32, u32)
    read_primitive!(read_u64, u64)
    read_primitive!(read_int, int)
    read_primitive!(read_i8, i8)
    read_primitive!(read_i16, i16)
    read_primitive!(read_i32, i32)
    read_primitive!(read_i64, i64)

    fn read_f32(&mut self) -> DecodeResult<f32> { self.read_f64().map(|x| x as f32) }

    fn read_f64(&mut self) -> DecodeResult<f64> {
        debug!("read_f64");
        match self.pop() {
            I64(f) => Ok(f as f64),
            U64(f) => Ok(f as f64),
            F64(f) => Ok(f),
            String(s) => {
                // re: #12967.. a type w/ numeric keys (ie HashMap<uint, V> etc)
                // is going to have a string here, as per JSON spec.
                match std::from_str::from_str(s.as_slice()) {
                    Some(f) => Ok(f),
                    None => Err(ExpectedError("Number".to_string(), s)),
                }
            },
            Null => Ok(f64::NAN),
            value => Err(ExpectedError("Number".to_string(), format!("{}", value)))
        }
    }

    fn read_bool(&mut self) -> DecodeResult<bool> {
        debug!("read_bool");
        expect!(self.pop(), Boolean)
    }

    fn read_char(&mut self) -> DecodeResult<char> {
        let s = try!(self.read_str());
        {
            let mut it = s.as_slice().chars();
            match (it.next(), it.next()) {
                // exactly one character
                (Some(c), None) => return Ok(c),
                _ => ()
            }
        }
        Err(ExpectedError("single character string".to_string(), format!("{}", s)))
    }

    fn read_str(&mut self) -> DecodeResult<string::String> {
        debug!("read_str");
        expect!(self.pop(), String)
    }

    fn read_enum<T>(&mut self,
                    name: &str,
                    f: |&mut Decoder| -> DecodeResult<T>) -> DecodeResult<T> {
        debug!("read_enum({})", name);
        f(self)
    }

    fn read_enum_variant<T>(&mut self,
                            names: &[&str],
                            f: |&mut Decoder, uint| -> DecodeResult<T>)
                            -> DecodeResult<T> {
        debug!("read_enum_variant(names={})", names);
        let name = match self.pop() {
            String(s) => s,
            Object(mut o) => {
                let n = match o.pop(&"variant".to_string()) {
                    Some(String(s)) => s,
                    Some(val) => {
                        return Err(ExpectedError("String".to_string(), format!("{}", val)))
                    }
                    None => {
                        return Err(MissingFieldError("variant".to_string()))
                    }
                };
                match o.pop(&"fields".to_string()) {
                    Some(List(l)) => {
                        for field in l.into_iter().rev() {
                            self.stack.push(field);
                        }
                    },
                    Some(val) => {
                        return Err(ExpectedError("List".to_string(), format!("{}", val)))
                    }
                    None => {
                        return Err(MissingFieldError("fields".to_string()))
                    }
                }
                n
            }
            json => {
                return Err(ExpectedError("String or Object".to_string(), format!("{}", json)))
            }
        };
        let idx = match names.iter()
                             .position(|n| str::eq_slice(*n, name.as_slice())) {
            Some(idx) => idx,
            None => return Err(UnknownVariantError(name))
        };
        f(self, idx)
    }

    fn read_enum_variant_arg<T>(&mut self, idx: uint, f: |&mut Decoder| -> DecodeResult<T>)
                                -> DecodeResult<T> {
        debug!("read_enum_variant_arg(idx={})", idx);
        f(self)
    }

    fn read_enum_struct_variant<T>(&mut self,
                                   names: &[&str],
                                   f: |&mut Decoder, uint| -> DecodeResult<T>)
                                   -> DecodeResult<T> {
        debug!("read_enum_struct_variant(names={})", names);
        self.read_enum_variant(names, f)
    }


    fn read_enum_struct_variant_field<T>(&mut self,
                                         name: &str,
                                         idx: uint,
                                         f: |&mut Decoder| -> DecodeResult<T>)
                                         -> DecodeResult<T> {
        debug!("read_enum_struct_variant_field(name={}, idx={})", name, idx);
        self.read_enum_variant_arg(idx, f)
    }

    fn read_struct<T>(&mut self,
                      name: &str,
                      len: uint,
                      f: |&mut Decoder| -> DecodeResult<T>)
                      -> DecodeResult<T> {
        debug!("read_struct(name={}, len={})", name, len);
        let value = try!(f(self));
        self.pop();
        Ok(value)
    }

    fn read_struct_field<T>(&mut self,
                            name: &str,
                            idx: uint,
                            f: |&mut Decoder| -> DecodeResult<T>)
                            -> DecodeResult<T> {
        debug!("read_struct_field(name={}, idx={})", name, idx);
        let mut obj = try!(expect!(self.pop(), Object));

        let value = match obj.pop(&name.to_string()) {
            None => {
                // Add a Null and try to parse it as an Option<_>
                // to get None as a default value.
                self.stack.push(Null);
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
        self.stack.push(Object(obj));
        Ok(value)
    }

    fn read_tuple<T>(&mut self, f: |&mut Decoder, uint| -> DecodeResult<T>) -> DecodeResult<T> {
        debug!("read_tuple()");
        self.read_seq(f)
    }

    fn read_tuple_arg<T>(&mut self,
                         idx: uint,
                         f: |&mut Decoder| -> DecodeResult<T>) -> DecodeResult<T> {
        debug!("read_tuple_arg(idx={})", idx);
        self.read_seq_elt(idx, f)
    }

    fn read_tuple_struct<T>(&mut self,
                            name: &str,
                            f: |&mut Decoder, uint| -> DecodeResult<T>)
                            -> DecodeResult<T> {
        debug!("read_tuple_struct(name={})", name);
        self.read_tuple(f)
    }

    fn read_tuple_struct_arg<T>(&mut self,
                                idx: uint,
                                f: |&mut Decoder| -> DecodeResult<T>)
                                -> DecodeResult<T> {
        debug!("read_tuple_struct_arg(idx={})", idx);
        self.read_tuple_arg(idx, f)
    }

    fn read_option<T>(&mut self, f: |&mut Decoder, bool| -> DecodeResult<T>) -> DecodeResult<T> {
        debug!("read_option()");
        match self.pop() {
            Null => f(self, false),
            value => { self.stack.push(value); f(self, true) }
        }
    }

    fn read_seq<T>(&mut self, f: |&mut Decoder, uint| -> DecodeResult<T>) -> DecodeResult<T> {
        debug!("read_seq()");
        let list = try!(expect!(self.pop(), List));
        let len = list.len();
        for v in list.into_iter().rev() {
            self.stack.push(v);
        }
        f(self, len)
    }

    fn read_seq_elt<T>(&mut self,
                       idx: uint,
                       f: |&mut Decoder| -> DecodeResult<T>) -> DecodeResult<T> {
        debug!("read_seq_elt(idx={})", idx);
        f(self)
    }

    fn read_map<T>(&mut self, f: |&mut Decoder, uint| -> DecodeResult<T>) -> DecodeResult<T> {
        debug!("read_map()");
        let obj = try!(expect!(self.pop(), Object));
        let len = obj.len();
        for (key, value) in obj.into_iter() {
            self.stack.push(value);
            self.stack.push(String(key));
        }
        f(self, len)
    }

    fn read_map_elt_key<T>(&mut self, idx: uint, f: |&mut Decoder| -> DecodeResult<T>)
                           -> DecodeResult<T> {
        debug!("read_map_elt_key(idx={})", idx);
        f(self)
    }

    fn read_map_elt_val<T>(&mut self, idx: uint, f: |&mut Decoder| -> DecodeResult<T>)
                           -> DecodeResult<T> {
        debug!("read_map_elt_val(idx={})", idx);
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

macro_rules! to_json_impl_i64(
    ($($t:ty), +) => (
        $(impl ToJson for $t {
            fn to_json(&self) -> Json { I64(*self as i64) }
        })+
    )
)

to_json_impl_i64!(int, i8, i16, i32, i64)

macro_rules! to_json_impl_u64(
    ($($t:ty), +) => (
        $(impl ToJson for $t {
            fn to_json(&self) -> Json { U64(*self as u64) }
        })+
    )
)

to_json_impl_u64!(uint, u8, u16, u32, u64)

impl ToJson for Json {
    fn to_json(&self) -> Json { self.clone() }
}

impl ToJson for f32 {
    fn to_json(&self) -> Json { (*self as f64).to_json() }
}

impl ToJson for f64 {
    fn to_json(&self) -> Json {
        match self.classify() {
            FPNaN | FPInfinite => Null,
            _                  => F64(*self)
        }
    }
}

impl ToJson for () {
    fn to_json(&self) -> Json { Null }
}

impl ToJson for bool {
    fn to_json(&self) -> Json { Boolean(*self) }
}

impl ToJson for string::String {
    fn to_json(&self) -> Json { String((*self).clone()) }
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
                    ($(ref $tyvar),*,) => List(vec![$($tyvar.to_json()),*])
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

impl<'a, A: ToJson> ToJson for &'a [A] {
    fn to_json(&self) -> Json { List(self.iter().map(|elt| elt.to_json()).collect()) }
}

impl<A: ToJson> ToJson for Vec<A> {
    fn to_json(&self) -> Json { List(self.iter().map(|elt| elt.to_json()).collect()) }
}

impl<A: ToJson> ToJson for TreeMap<string::String, A> {
    fn to_json(&self) -> Json {
        let mut d = TreeMap::new();
        for (key, value) in self.iter() {
            d.insert((*key).clone(), value.to_json());
        }
        Object(d)
    }
}

impl<A: ToJson> ToJson for HashMap<string::String, A> {
    fn to_json(&self) -> Json {
        let mut d = TreeMap::new();
        for (key, value) in self.iter() {
            d.insert((*key).clone(), value.to_json());
        }
        Object(d)
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

impl fmt::Show for Json {
    /// Encodes a json value into a string
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.to_writer(f).map_err(|_| fmt::WriteError)
    }
}

impl std::from_str::FromStr for Json {
    fn from_str(s: &str) -> Option<Json> {
        from_str(s).ok()
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use {Encodable, Decodable};
    use super::{List, Encoder, Decoder, Error, Boolean, I64, U64, F64, String, Null,
                PrettyEncoder, Object, Json, from_str, ParseError, ExpectedError,
                MissingFieldError, UnknownVariantError, DecodeResult, DecoderError,
                JsonEvent, Parser, StackElement,
                ObjectStart, ObjectEnd, ListStart, ListEnd, BooleanValue, U64Value,
                F64Value, StringValue, NullValue, SyntaxError, Key, Index, Stack,
                InvalidSyntax, InvalidNumber, EOFWhileParsingObject, EOFWhileParsingList,
                EOFWhileParsingValue, EOFWhileParsingString, KeyMustBeAString, ExpectedColon,
                TrailingCharacters, TrailingComma};
    use std::{i64, u64, f32, f64, io};
    use std::collections::TreeMap;
    use std::string;

    #[deriving(Decodable, Eq, PartialEq, Show)]
    struct OptionData {
        opt: Option<uint>,
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
        assert_eq!(obj, OptionData { opt: Some(10u) });
    }

    #[test]
    fn test_decode_option_malformed() {
        check_err::<OptionData>("{ \"opt\": [] }",
                                ExpectedError("Number".to_string(), "[]".to_string()));
        check_err::<OptionData>("{ \"opt\": false }",
                                ExpectedError("Number".to_string(), "false".to_string()));
    }

    #[deriving(PartialEq, Encodable, Decodable, Show)]
    enum Animal {
        Dog,
        Frog(string::String, int)
    }

    #[deriving(PartialEq, Encodable, Decodable, Show)]
    struct Inner {
        a: (),
        b: uint,
        c: Vec<string::String>,
    }

    #[deriving(PartialEq, Encodable, Decodable, Show)]
    struct Outer {
        inner: Vec<Inner>,
    }

    fn mk_object(items: &[(string::String, Json)]) -> Json {
        let mut d = TreeMap::new();

        for item in items.iter() {
            match *item {
                (ref key, ref value) => { d.insert((*key).clone(), (*value).clone()); },
            }
        };

        Object(d)
    }

    #[test]
    fn test_from_str_trait() {
        let s = "null";
        assert!(::std::from_str::from_str::<Json>(s).unwrap() == from_str(s).unwrap());
    }

    #[test]
    fn test_write_null() {
        assert_eq!(Null.to_string().into_string(), "null".to_string());
        assert_eq!(Null.to_pretty_str().into_string(), "null".to_string());
    }

    #[test]
    fn test_write_i64() {
        assert_eq!(U64(0).to_string().into_string(), "0".to_string());
        assert_eq!(U64(0).to_pretty_str().into_string(), "0".to_string());

        assert_eq!(U64(1234).to_string().into_string(), "1234".to_string());
        assert_eq!(U64(1234).to_pretty_str().into_string(), "1234".to_string());

        assert_eq!(I64(-5678).to_string().into_string(), "-5678".to_string());
        assert_eq!(I64(-5678).to_pretty_str().into_string(), "-5678".to_string());
    }

    #[test]
    fn test_write_f64() {
        assert_eq!(F64(3.0).to_string().into_string(), "3".to_string());
        assert_eq!(F64(3.0).to_pretty_str().into_string(), "3".to_string());

        assert_eq!(F64(3.1).to_string().into_string(), "3.1".to_string());
        assert_eq!(F64(3.1).to_pretty_str().into_string(), "3.1".to_string());

        assert_eq!(F64(-1.5).to_string().into_string(), "-1.5".to_string());
        assert_eq!(F64(-1.5).to_pretty_str().into_string(), "-1.5".to_string());

        assert_eq!(F64(0.5).to_string().into_string(), "0.5".to_string());
        assert_eq!(F64(0.5).to_pretty_str().into_string(), "0.5".to_string());

        assert_eq!(F64(f64::NAN).to_string().into_string(), "null".to_string());
        assert_eq!(F64(f64::NAN).to_pretty_str().into_string(), "null".to_string());

        assert_eq!(F64(f64::INFINITY).to_string().into_string(), "null".to_string());
        assert_eq!(F64(f64::INFINITY).to_pretty_str().into_string(), "null".to_string());

        assert_eq!(F64(f64::NEG_INFINITY).to_string().into_string(), "null".to_string());
        assert_eq!(F64(f64::NEG_INFINITY).to_pretty_str().into_string(), "null".to_string());
    }

    #[test]
    fn test_write_str() {
        assert_eq!(String("".to_string()).to_string().into_string(), "\"\"".to_string());
        assert_eq!(String("".to_string()).to_pretty_str().into_string(), "\"\"".to_string());

        assert_eq!(String("foo".to_string()).to_string().into_string(), "\"foo\"".to_string());
        assert_eq!(String("foo".to_string()).to_pretty_str().into_string(), "\"foo\"".to_string());
    }

    #[test]
    fn test_write_bool() {
        assert_eq!(Boolean(true).to_string().into_string(), "true".to_string());
        assert_eq!(Boolean(true).to_pretty_str().into_string(), "true".to_string());

        assert_eq!(Boolean(false).to_string().into_string(), "false".to_string());
        assert_eq!(Boolean(false).to_pretty_str().into_string(), "false".to_string());
    }

    #[test]
    fn test_write_list() {
        assert_eq!(List(vec![]).to_string().into_string(), "[]".to_string());
        assert_eq!(List(vec![]).to_pretty_str().into_string(), "[]".to_string());

        assert_eq!(List(vec![Boolean(true)]).to_string().into_string(), "[true]".to_string());
        assert_eq!(
            List(vec![Boolean(true)]).to_pretty_str().into_string(),
            "\
            [\n  \
                true\n\
            ]".to_string()
        );

        let long_test_list = List(vec![
            Boolean(false),
            Null,
            List(vec![String("foo\nbar".to_string()), F64(3.5)])]);

        assert_eq!(long_test_list.to_string().into_string(),
            "[false,null,[\"foo\\nbar\",3.5]]".to_string());
        assert_eq!(
            long_test_list.to_pretty_str().into_string(),
            "\
            [\n  \
                false,\n  \
                null,\n  \
                [\n    \
                    \"foo\\nbar\",\n    \
                    3.5\n  \
                ]\n\
            ]".to_string()
        );
    }

    #[test]
    fn test_write_object() {
        assert_eq!(mk_object([]).to_string().into_string(), "{}".to_string());
        assert_eq!(mk_object([]).to_pretty_str().into_string(), "{}".to_string());

        assert_eq!(
            mk_object([
                ("a".to_string(), Boolean(true))
            ]).to_string().into_string(),
            "{\"a\":true}".to_string()
        );
        assert_eq!(
            mk_object([("a".to_string(), Boolean(true))]).to_pretty_str(),
            "\
            {\n  \
                \"a\": true\n\
            }".to_string()
        );

        let complex_obj = mk_object([
                ("b".to_string(), List(vec![
                    mk_object([("c".to_string(), String("\x0c\r".to_string()))]),
                    mk_object([("d".to_string(), String("".to_string()))])
                ]))
            ]);

        assert_eq!(
            complex_obj.to_string().into_string(),
            "{\
                \"b\":[\
                    {\"c\":\"\\f\\r\"},\
                    {\"d\":\"\"}\
                ]\
            }".to_string()
        );
        assert_eq!(
            complex_obj.to_pretty_str().into_string(),
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
            }".to_string()
        );

        let a = mk_object([
            ("a".to_string(), Boolean(true)),
            ("b".to_string(), List(vec![
                mk_object([("c".to_string(), String("\x0c\r".to_string()))]),
                mk_object([("d".to_string(), String("".to_string()))])
            ]))
        ]);

        // We can't compare the strings directly because the object fields be
        // printed in a different order.
        assert_eq!(a.clone(), from_str(a.to_string().as_slice()).unwrap());
        assert_eq!(a.clone(),
                   from_str(a.to_pretty_str().as_slice()).unwrap());
    }

    fn with_str_writer(f: |&mut io::Writer|) -> string::String {
        use std::io::MemWriter;
        use std::str;

        let mut m = MemWriter::new();
        f(&mut m as &mut io::Writer);
        str::from_utf8(m.unwrap().as_slice()).unwrap().to_string()
    }

    #[test]
    fn test_write_enum() {
        let animal = Dog;
        assert_eq!(
            with_str_writer(|writer| {
                let mut encoder = Encoder::new(writer);
                animal.encode(&mut encoder).unwrap();
            }),
            "\"Dog\"".to_string()
        );
        assert_eq!(
            with_str_writer(|writer| {
                let mut encoder = PrettyEncoder::new(writer);
                animal.encode(&mut encoder).unwrap();
            }),
            "\"Dog\"".to_string()
        );

        let animal = Frog("Henry".to_string(), 349);
        assert_eq!(
            with_str_writer(|writer| {
                let mut encoder = Encoder::new(writer);
                animal.encode(&mut encoder).unwrap();
            }),
            "{\"variant\":\"Frog\",\"fields\":[\"Henry\",349]}".to_string()
        );
        assert_eq!(
            with_str_writer(|writer| {
                let mut encoder = PrettyEncoder::new(writer);
                animal.encode(&mut encoder).unwrap();
            }),
            "\
            [\n  \
                \"Frog\",\n  \
                \"Henry\",\n  \
                349\n\
            ]".to_string()
        );
    }

    #[test]
    fn test_write_some() {
        let value = Some("jodhpurs".to_string());
        let s = with_str_writer(|writer| {
            let mut encoder = Encoder::new(writer);
            value.encode(&mut encoder).unwrap();
        });
        assert_eq!(s, "\"jodhpurs\"".to_string());

        let value = Some("jodhpurs".to_string());
        let s = with_str_writer(|writer| {
            let mut encoder = PrettyEncoder::new(writer);
            value.encode(&mut encoder).unwrap();
        });
        assert_eq!(s, "\"jodhpurs\"".to_string());
    }

    #[test]
    fn test_write_none() {
        let value: Option<string::String> = None;
        let s = with_str_writer(|writer| {
            let mut encoder = Encoder::new(writer);
            value.encode(&mut encoder).unwrap();
        });
        assert_eq!(s, "null".to_string());

        let s = with_str_writer(|writer| {
            let mut encoder = Encoder::new(writer);
            value.encode(&mut encoder).unwrap();
        });
        assert_eq!(s, "null".to_string());
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
        assert_eq!(from_str("\"\\u12ab\""), Ok(String("\u12ab".to_string())));
        assert_eq!(from_str("\"\\uAB12\""), Ok(String("\uAB12".to_string())));
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
                 ("\"\\u12ab\"", "\u12ab"),
                 ("\"\\uAB12\"", "\uAB12")];

        for &(i, o) in s.iter() {
            let v: string::String = super::decode(i).unwrap();
            assert_eq!(v.as_slice(), o);
        }
    }

    #[test]
    fn test_read_list() {
        assert_eq!(from_str("["),     Err(SyntaxError(EOFWhileParsingValue, 1, 2)));
        assert_eq!(from_str("[1"),    Err(SyntaxError(EOFWhileParsingList,  1, 3)));
        assert_eq!(from_str("[1,"),   Err(SyntaxError(EOFWhileParsingValue, 1, 4)));
        assert_eq!(from_str("[1,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
        assert_eq!(from_str("[6 7]"), Err(SyntaxError(InvalidSyntax,        1, 4)));

        assert_eq!(from_str("[]"), Ok(List(vec![])));
        assert_eq!(from_str("[ ]"), Ok(List(vec![])));
        assert_eq!(from_str("[true]"), Ok(List(vec![Boolean(true)])));
        assert_eq!(from_str("[ false ]"), Ok(List(vec![Boolean(false)])));
        assert_eq!(from_str("[null]"), Ok(List(vec![Null])));
        assert_eq!(from_str("[3, 1]"),
                     Ok(List(vec![U64(3), U64(1)])));
        assert_eq!(from_str("\n[3, 2]\n"),
                     Ok(List(vec![U64(3), U64(2)])));
        assert_eq!(from_str("[2, [4, 1]]"),
               Ok(List(vec![U64(2), List(vec![U64(4), U64(1)])])));
    }

    #[test]
    fn test_decode_list() {
        let v: Vec<()> = super::decode("[]").unwrap();
        assert_eq!(v, vec![]);

        let v: Vec<()> = super::decode("[null]").unwrap();
        assert_eq!(v, vec![()]);

        let v: Vec<bool> = super::decode("[true]").unwrap();
        assert_eq!(v, vec![true]);

        let v: Vec<int> = super::decode("[3, 1]").unwrap();
        assert_eq!(v, vec![3, 1]);

        let v: Vec<Vec<uint>> = super::decode("[[3], [1, 2]]").unwrap();
        assert_eq!(v, vec![vec![3], vec![1, 2]]);
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

        assert_eq!(from_str("{}").unwrap(), mk_object([]));
        assert_eq!(from_str("{\"a\": 3}").unwrap(),
                  mk_object([("a".to_string(), U64(3))]));

        assert_eq!(from_str(
                      "{ \"a\": null, \"b\" : true }").unwrap(),
                  mk_object([
                      ("a".to_string(), Null),
                      ("b".to_string(), Boolean(true))]));
        assert_eq!(from_str("\n{ \"a\": null, \"b\" : true }\n").unwrap(),
                  mk_object([
                      ("a".to_string(), Null),
                      ("b".to_string(), Boolean(true))]));
        assert_eq!(from_str(
                      "{\"a\" : 1.0 ,\"b\": [ true ]}").unwrap(),
                  mk_object([
                      ("a".to_string(), F64(1.0)),
                      ("b".to_string(), List(vec![Boolean(true)]))
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
                  mk_object([
                      ("a".to_string(), F64(1.0)),
                      ("b".to_string(), List(vec![
                          Boolean(true),
                          String("foo\nbar".to_string()),
                          mk_object([
                              ("c".to_string(), mk_object([("d".to_string(), Null)]))
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

    #[deriving(Decodable)]
    struct FloatStruct {
        f: f64,
        a: Vec<f64>
    }
    #[test]
    fn test_decode_struct_with_nan() {
        let s = "{\"f\":null,\"a\":[null,123]}";
        let obj: FloatStruct = super::decode(s).unwrap();
        assert!(obj.f.is_nan());
        assert!(obj.a.get(0).is_nan());
        assert_eq!(obj.a.get(1), &123f64);
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
        let mut map: TreeMap<string::String, Animal> = super::decode(s).unwrap();

        assert_eq!(map.pop(&"a".to_string()), Some(Dog));
        assert_eq!(map.pop(&"b".to_string()), Some(Frog("Henry".to_string(), 349)));
    }

    #[test]
    fn test_multiline_errors() {
        assert_eq!(from_str("{\n  \"foo\":\n \"bar\""),
            Err(SyntaxError(EOFWhileParsingObject, 3u, 8u)));
    }

    #[deriving(Decodable)]
    #[allow(dead_code)]
    struct DecodeStruct {
        x: f64,
        y: bool,
        z: string::String,
        w: Vec<DecodeStruct>
    }
    #[deriving(Decodable)]
    enum DecodeEnum {
        A(f64),
        B(string::String)
    }
    fn check_err<T: Decodable<Decoder, DecoderError>>(to_parse: &'static str,
                                                      expected: DecoderError) {
        let res: DecodeResult<T> = match from_str(to_parse) {
            Err(e) => Err(ParseError(e)),
            Ok(json) => Decodable::decode(&mut Decoder::new(json))
        };
        match res {
            Ok(_) => fail!("`{}` parsed & decoded ok, expecting error `{}`",
                              to_parse, expected),
            Err(ParseError(e)) => fail!("`{}` is not valid json: {}",
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
                                  ExpectedError("List".to_string(), "null".to_string()));
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
                                ExpectedError("List".to_string(), "null".to_string()));
        check_err::<DecodeEnum>("{\"variant\": \"C\", \"fields\": []}",
                                UnknownVariantError("C".to_string()));
    }

    #[test]
    fn test_find(){
        let json_value = from_str("{\"dog\" : \"cat\"}").unwrap();
        let found_str = json_value.find(&"dog".to_string());
        assert!(found_str.is_some() && found_str.unwrap().as_string().unwrap() == "cat");
    }

    #[test]
    fn test_find_path(){
        let json_value = from_str("{\"dog\":{\"cat\": {\"mouse\" : \"cheese\"}}}").unwrap();
        let found_str = json_value.find_path(&[&"dog".to_string(),
                                             &"cat".to_string(), &"mouse".to_string()]);
        assert!(found_str.is_some() && found_str.unwrap().as_string().unwrap() == "cheese");
    }

    #[test]
    fn test_search(){
        let json_value = from_str("{\"dog\":{\"cat\": {\"mouse\" : \"cheese\"}}}").unwrap();
        let found_str = json_value.search(&"mouse".to_string()).and_then(|j| j.as_string());
        assert!(found_str.is_some());
        assert!(found_str.unwrap() == "cheese");
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
    fn test_is_list(){
        let json_value = from_str("[1, 2, 3]").unwrap();
        assert!(json_value.is_list());
    }

    #[test]
    fn test_as_list(){
        let json_value = from_str("[1, 2, 3]").unwrap();
        let json_list = json_value.as_list();
        let expected_length = 3;
        assert!(json_list.is_some() && json_list.unwrap().len() == expected_length);
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
        use std::io::Writer;
        use std::io::MemWriter;
        use std::collections::HashMap;
        let mut hm: HashMap<uint, bool> = HashMap::new();
        hm.insert(1, true);
        let mut mem_buf = MemWriter::new();
        {
            let mut encoder = Encoder::new(&mut mem_buf as &mut io::Writer);
            hm.encode(&mut encoder).unwrap();
        }
        let bytes = mem_buf.unwrap();
        let json_str = from_utf8(bytes.as_slice()).unwrap();
        match from_str(json_str) {
            Err(_) => fail!("Unable to parse json_str: {}", json_str),
            _ => {} // it parsed and we are good to go
        }
    }

    #[test]
    fn test_prettyencode_hashmap_with_numeric_key() {
        use std::str::from_utf8;
        use std::io::Writer;
        use std::io::MemWriter;
        use std::collections::HashMap;
        let mut hm: HashMap<uint, bool> = HashMap::new();
        hm.insert(1, true);
        let mut mem_buf = MemWriter::new();
        {
            let mut encoder = PrettyEncoder::new(&mut mem_buf as &mut io::Writer);
            hm.encode(&mut encoder).unwrap()
        }
        let bytes = mem_buf.unwrap();
        let json_str = from_utf8(bytes.as_slice()).unwrap();
        match from_str(json_str) {
            Err(_) => fail!("Unable to parse json_str: {}", json_str),
            _ => {} // it parsed and we are good to go
        }
    }

    #[test]
    fn test_prettyencoder_indent_level_param() {
        use std::str::from_utf8;
        use std::io::MemWriter;
        use std::collections::TreeMap;

        let mut tree = TreeMap::new();

        tree.insert("hello".into_string(), String("guten tag".into_string()));
        tree.insert("goodbye".into_string(), String("sayonara".into_string()));

        let json = List(
            // The following layout below should look a lot like
            // the pretty-printed JSON (indent * x)
            vec!
            ( // 0x
                String("greetings".into_string()), // 1x
                Object(tree), // 1x + 2x + 2x + 1x
            ) // 0x
            // End JSON list (7 lines)
        );

        // Helper function for counting indents
        fn indents(source: &str) -> uint {
            let trimmed = source.trim_left_chars(' ');
            source.len() - trimmed.len()
        }

        // Test up to 4 spaces of indents (more?)
        for i in range(0, 4u) {
            let mut writer = MemWriter::new();
            {
                let ref mut encoder = PrettyEncoder::new(&mut writer);
                encoder.set_indent(i);
                json.encode(encoder).unwrap();
            }

            let bytes = writer.unwrap();
            let printed = from_utf8(bytes.as_slice()).unwrap();

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
    fn test_hashmap_with_numeric_key_can_handle_double_quote_delimited_key() {
        use std::collections::HashMap;
        use Decodable;
        let json_str = "{\"1\":true}";
        let json_obj = match from_str(json_str) {
            Err(_) => fail!("Unable to parse json_str: {}", json_str),
            Ok(o) => o
        };
        let mut decoder = Decoder::new(json_obj);
        let _hm: HashMap<uint, bool> = Decodable::decode(&mut decoder).unwrap();
    }

    #[test]
    fn test_hashmap_with_numeric_key_will_error_with_string_keys() {
        use std::collections::HashMap;
        use Decodable;
        let json_str = "{\"a\":true}";
        let json_obj = match from_str(json_str) {
            Err(_) => fail!("Unable to parse json_str: {}", json_str),
            Ok(o) => o
        };
        let mut decoder = Decoder::new(json_obj);
        let result: Result<HashMap<uint, bool>, DecoderError> = Decodable::decode(&mut decoder);
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
            if !parser.stack().is_equal_to(expected_stack.as_slice()) {
                fail!("Parser stack is not equal to {}", expected_stack);
            }
            assert_eq!(&evt, expected_evt);
            i+=1;
        }
    }
    #[test]
    #[cfg_attr(target_word_size = "32", ignore)] // FIXME(#14064)
    fn test_streaming_parser() {
        assert_stream_equal(
            r#"{ "foo":"bar", "array" : [0, 1, 2, 3, 4, 5], "idents":[null,true,false]}"#,
            vec![
                (ObjectStart,             vec![]),
                  (StringValue("bar".to_string()),   vec![Key("foo")]),
                  (ListStart,             vec![Key("array")]),
                    (U64Value(0),         vec![Key("array"), Index(0)]),
                    (U64Value(1),         vec![Key("array"), Index(1)]),
                    (U64Value(2),         vec![Key("array"), Index(2)]),
                    (U64Value(3),         vec![Key("array"), Index(3)]),
                    (U64Value(4),         vec![Key("array"), Index(4)]),
                    (U64Value(5),         vec![Key("array"), Index(5)]),
                  (ListEnd,               vec![Key("array")]),
                  (ListStart,             vec![Key("idents")]),
                    (NullValue,           vec![Key("idents"), Index(0)]),
                    (BooleanValue(true),  vec![Key("idents"), Index(1)]),
                    (BooleanValue(false), vec![Key("idents"), Index(2)]),
                  (ListEnd,               vec![Key("idents")]),
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
    #[cfg_attr(target_word_size = "32", ignore)] // FIXME(#14064)
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
                  (U64Value(3),      vec![Key("a")]),
                (ObjectEnd,          vec![]),
            ]
        );
        assert_stream_equal(
            "{ \"a\": null, \"b\" : true }",
            vec![
                (ObjectStart,           vec![]),
                  (NullValue,           vec![Key("a")]),
                  (BooleanValue(true),  vec![Key("b")]),
                (ObjectEnd,             vec![]),
            ]
        );
        assert_stream_equal(
            "{\"a\" : 1.0 ,\"b\": [ true ]}",
            vec![
                (ObjectStart,           vec![]),
                  (F64Value(1.0),       vec![Key("a")]),
                  (ListStart,           vec![Key("b")]),
                    (BooleanValue(true),vec![Key("b"), Index(0)]),
                  (ListEnd,             vec![Key("b")]),
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
                  (F64Value(1.0),               vec![Key("a")]),
                  (ListStart,                   vec![Key("b")]),
                    (BooleanValue(true),        vec![Key("b"), Index(0)]),
                    (StringValue("foo\nbar".to_string()),  vec![Key("b"), Index(1)]),
                    (ObjectStart,               vec![Key("b"), Index(2)]),
                      (ObjectStart,             vec![Key("b"), Index(2), Key("c")]),
                        (NullValue,             vec![Key("b"), Index(2), Key("c"), Key("d")]),
                      (ObjectEnd,               vec![Key("b"), Index(2), Key("c")]),
                    (ObjectEnd,                 vec![Key("b"), Index(2)]),
                  (ListEnd,                     vec![Key("b")]),
                (ObjectEnd,                     vec![]),
            ]
        );
    }
    #[test]
    #[cfg_attr(target_word_size = "32", ignore)] // FIXME(#14064)
    fn test_read_list_streaming() {
        assert_stream_equal(
            "[]",
            vec![
                (ListStart, vec![]),
                (ListEnd,   vec![]),
            ]
        );
        assert_stream_equal(
            "[ ]",
            vec![
                (ListStart, vec![]),
                (ListEnd,   vec![]),
            ]
        );
        assert_stream_equal(
            "[true]",
            vec![
                (ListStart,              vec![]),
                    (BooleanValue(true), vec![Index(0)]),
                (ListEnd,                vec![]),
            ]
        );
        assert_stream_equal(
            "[ false ]",
            vec![
                (ListStart,               vec![]),
                    (BooleanValue(false), vec![Index(0)]),
                (ListEnd,                 vec![]),
            ]
        );
        assert_stream_equal(
            "[null]",
            vec![
                (ListStart,     vec![]),
                    (NullValue, vec![Index(0)]),
                (ListEnd,       vec![]),
            ]
        );
        assert_stream_equal(
            "[3, 1]",
            vec![
                (ListStart,     vec![]),
                    (U64Value(3), vec![Index(0)]),
                    (U64Value(1), vec![Index(1)]),
                (ListEnd,       vec![]),
            ]
        );
        assert_stream_equal(
            "\n[3, 2]\n",
            vec![
                (ListStart,     vec![]),
                    (U64Value(3), vec![Index(0)]),
                    (U64Value(2), vec![Index(1)]),
                (ListEnd,       vec![]),
            ]
        );
        assert_stream_equal(
            "[2, [4, 1]]",
            vec![
                (ListStart,            vec![]),
                    (U64Value(2),      vec![Index(0)]),
                    (ListStart,        vec![Index(1)]),
                        (U64Value(4),  vec![Index(1), Index(0)]),
                        (U64Value(1),  vec![Index(1), Index(1)]),
                    (ListEnd,          vec![Index(1)]),
                (ListEnd,              vec![]),
            ]
        );

        assert_eq!(last_event("["), Error(SyntaxError(EOFWhileParsingValue, 1,  2)));

        assert_eq!(from_str("["),     Err(SyntaxError(EOFWhileParsingValue, 1, 2)));
        assert_eq!(from_str("[1"),    Err(SyntaxError(EOFWhileParsingList,  1, 3)));
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
        assert!(stack.len() == 0);
        assert!(!stack.last_is_index());

        stack.push_index(0);
        stack.bump_index();

        assert!(stack.len() == 1);
        assert!(stack.is_equal_to([Index(1)]));
        assert!(stack.starts_with([Index(1)]));
        assert!(stack.ends_with([Index(1)]));
        assert!(stack.last_is_index());
        assert!(stack.get(0) == Index(1));

        stack.push_key("foo".to_string());

        assert!(stack.len() == 2);
        assert!(stack.is_equal_to([Index(1), Key("foo")]));
        assert!(stack.starts_with([Index(1), Key("foo")]));
        assert!(stack.starts_with([Index(1)]));
        assert!(stack.ends_with([Index(1), Key("foo")]));
        assert!(stack.ends_with([Key("foo")]));
        assert!(!stack.last_is_index());
        assert!(stack.get(0) == Index(1));
        assert!(stack.get(1) == Key("foo"));

        stack.push_key("bar".to_string());

        assert!(stack.len() == 3);
        assert!(stack.is_equal_to([Index(1), Key("foo"), Key("bar")]));
        assert!(stack.starts_with([Index(1)]));
        assert!(stack.starts_with([Index(1), Key("foo")]));
        assert!(stack.starts_with([Index(1), Key("foo"), Key("bar")]));
        assert!(stack.ends_with([Key("bar")]));
        assert!(stack.ends_with([Key("foo"), Key("bar")]));
        assert!(stack.ends_with([Index(1), Key("foo"), Key("bar")]));
        assert!(!stack.last_is_index());
        assert!(stack.get(0) == Index(1));
        assert!(stack.get(1) == Key("foo"));
        assert!(stack.get(2) == Key("bar"));

        stack.pop();

        assert!(stack.len() == 2);
        assert!(stack.is_equal_to([Index(1), Key("foo")]));
        assert!(stack.starts_with([Index(1), Key("foo")]));
        assert!(stack.starts_with([Index(1)]));
        assert!(stack.ends_with([Index(1), Key("foo")]));
        assert!(stack.ends_with([Key("foo")]));
        assert!(!stack.last_is_index());
        assert!(stack.get(0) == Index(1));
        assert!(stack.get(1) == Key("foo"));
    }

    #[test]
    fn test_to_json() {
        use std::collections::{HashMap,TreeMap};
        use super::ToJson;

        let list2 = List(vec!(U64(1), U64(2)));
        let list3 = List(vec!(U64(1), U64(2), U64(3)));
        let object = {
            let mut tree_map = TreeMap::new();
            tree_map.insert("a".to_string(), U64(1));
            tree_map.insert("b".to_string(), U64(2));
            Object(tree_map)
        };

        assert_eq!(list2.to_json(), list2);
        assert_eq!(object.to_json(), object);
        assert_eq!(3_i.to_json(), I64(3));
        assert_eq!(4_i8.to_json(), I64(4));
        assert_eq!(5_i16.to_json(), I64(5));
        assert_eq!(6_i32.to_json(), I64(6));
        assert_eq!(7_i64.to_json(), I64(7));
        assert_eq!(8_u.to_json(), U64(8));
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
        assert_eq!("abc".to_string().to_json(), String("abc".to_string()));
        assert_eq!((1u, 2u).to_json(), list2);
        assert_eq!((1u, 2u, 3u).to_json(), list3);
        assert_eq!([1u, 2].to_json(), list2);
        assert_eq!((&[1u, 2, 3]).to_json(), list3);
        assert_eq!((vec![1u, 2]).to_json(), list2);
        assert_eq!(vec!(1u, 2, 3).to_json(), list3);
        let mut tree_map = TreeMap::new();
        tree_map.insert("a".to_string(), 1u);
        tree_map.insert("b".to_string(), 2);
        assert_eq!(tree_map.to_json(), object);
        let mut hash_map = HashMap::new();
        hash_map.insert("a".to_string(), 1u);
        hash_map.insert("b".to_string(), 2);
        assert_eq!(hash_map.to_json(), object);
        assert_eq!(Some(15i).to_json(), I64(15));
        assert_eq!(Some(15u).to_json(), U64(15));
        assert_eq!(None::<int>.to_json(), Null);
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
        for _ in range(0i, 500) {
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
            let mut parser = Parser::new(src.as_slice().chars());
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
        b.iter( || { let _ = from_str(src.as_slice()); });
    }
}
