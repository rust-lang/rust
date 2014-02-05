// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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
#[allow(missing_doc)];

/*!
JSON parsing and serialization

# What is JSON?

JSON (JavaScript Object Notation) is a way to write data in Javascript.
Like XML it allows one to encode structured data in a text format that can be read by humans easily.
Its native compatibility with JavaScript and its simple syntax make it used widely.

Json data are encoded in a form of "key":"value".
Data types that can be encoded are JavaScript types :
boolean (`true` or `false`), number (`f64`), string, array, object, null.
An object is a series of string keys mapping to values, in `"key": value` format.
Arrays are enclosed in square brackets ([ ... ]) and objects in curly brackets ({ ... }).
A simple JSON document encoding a person, his/her age, address and phone numbers could look like:

```
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

Rust provides a mechanism for low boilerplate encoding & decoding
of values to and from JSON via the serialization API.
To be able to encode a piece of data, it must implement the `serialize::Encodable` trait.
To be able to decode a piece of data, it must implement the `serialize::Decodable` trait.
The Rust compiler provides an annotation to automatically generate
the code for these traits: `#[deriving(Decodable, Encodable)]`

To encode using Encodable :

```rust
extern mod serialize;
use extra::json;
use std::io;
use serialize::Encodable;

 #[deriving(Encodable)]
 pub struct TestStruct   {
    data_str: ~str,
 }

fn main() {
    let to_encode_object = TestStruct{data_str:~"example of string to encode"};
    let mut m = io::MemWriter::new();
    {
        let mut encoder = json::Encoder::new(&mut m as &mut std::io::Writer);
        to_encode_object.encode(&mut encoder);
    }
}
```

Two wrapper functions are provided to encode a Encodable object
into a string (~str) or buffer (~[u8]): `str_encode(&m)` and `buffer_encode(&m)`.

```rust
use extra::json;
let to_encode_object = ~"example of string to encode";
let encoded_str: ~str = json::Encoder::str_encode(&to_encode_object);
```

JSON API provide an enum `json::Json` and a trait `ToJson` to encode object.
The trait `ToJson` encode object into a container `json::Json` and the API provide writer
to encode them into a stream or a string ...

When using `ToJson` the `Encodable` trait implementation is not mandatory.

A basic `ToJson` example using a TreeMap of attribute name / attribute value:


```rust
use extra::json;
use extra::json::ToJson;
use extra::treemap::TreeMap;

pub struct MyStruct  {
    attr1: u8,
    attr2: ~str,
}

impl ToJson for MyStruct {
    fn to_json( &self ) -> json::Json {
        let mut d = ~TreeMap::new();
        d.insert(~"attr1", self.attr1.to_json());
        d.insert(~"attr2", self.attr2.to_json());
        json::Object(d)
    }
}

fn main() {
    let test2: MyStruct = MyStruct {attr1: 1, attr2:~"test"};
    let tjson: json::Json = test2.to_json();
    let json_str: ~str = tjson.to_str();
}
```

To decode a json string using `Decodable` trait :

```rust
extern mod serialize;
use serialize::Decodable;

#[deriving(Decodable)]
pub struct MyStruct  {
     attr1: u8,
     attr2: ~str,
}

fn main() {
    let json_str_to_decode: ~str =
            ~"{\"attr1\":1,\"attr2\":\"toto\"}";
    let json_object = extra::json::from_str(json_str_to_decode);
    let mut decoder = extra::json::Decoder::new(json_object.unwrap());
    let decoded_object: MyStruct = Decodable::decode(&mut decoder); // create the final object
}
```

# Examples of use

## Using Autoserialization

Create a struct called TestStruct1 and serialize and deserialize it to and from JSON
using the serialization API, using the derived serialization code.

```rust
extern mod serialize;
use extra::json;
use serialize::{Encodable, Decodable};

 #[deriving(Decodable, Encodable)] //generate Decodable, Encodable impl.
 pub struct TestStruct1  {
    data_int: u8,
    data_str: ~str,
    data_vector: ~[u8],
 }

// To serialize use the `json::str_encode` to encode an object in a string.
// It calls the generated `Encodable` impl.
fn main() {
    let to_encode_object = TestStruct1
         {data_int: 1, data_str:~"toto", data_vector:~[2,3,4,5]};
    let encoded_str: ~str = json::Encoder::str_encode(&to_encode_object);

    // To unserialize use the `extra::json::from_str` and `extra::json::Decoder`

    let json_object = extra::json::from_str(encoded_str);
    let mut decoder = json::Decoder::new(json_object.unwrap());
    let decoded1: TestStruct1 = Decodable::decode(&mut decoder); // create the final object
}
```

## Using `ToJson`

This example use the ToJson impl to unserialize the json string.
Example of `ToJson` trait implementation for TestStruct1.

```rust
extern mod serialize;
use extra::json;
use extra::json::ToJson;
use serialize::{Encodable, Decodable};
use extra::treemap::TreeMap;

#[deriving(Decodable, Encodable)] // generate Decodable, Encodable impl.
pub struct TestStruct1  {
    data_int: u8,
    data_str: ~str,
    data_vector: ~[u8],
}

impl ToJson for TestStruct1 {
    fn to_json( &self ) -> json::Json {
        let mut d = ~TreeMap::new();
        d.insert(~"data_int", self.data_int.to_json());
        d.insert(~"data_str", self.data_str.to_json());
        d.insert(~"data_vector", self.data_vector.to_json());
        json::Object(d)
    }
}

fn main() {
    // Seralization using our impl of to_json

    let test2: TestStruct1 = TestStruct1 {data_int: 1, data_str:~"toto", data_vector:~[2,3,4,5]};
    let tjson: json::Json = test2.to_json();
    let json_str: ~str = tjson.to_str();

    // Unserialize like before.

    let mut decoder = json::Decoder::new(json::from_str(json_str).unwrap());
    // create the final object
    let decoded2: TestStruct1 = Decodable::decode(&mut decoder);
}
```

*/

use std::char;
use std::cast::transmute;
use std::f64;
use std::hashmap::HashMap;
use std::io;
use std::io::MemWriter;
use std::num;
use std::str;
use std::to_str;

use serialize::Encodable;
use serialize;
use treemap::TreeMap;

macro_rules! if_ok( ($e:expr) => (
    match $e { Ok(e) => e, Err(e) => { self.error = Err(e); return } }
) )

/// Represents a json value
#[deriving(Clone, Eq)]
pub enum Json {
    Number(f64),
    String(~str),
    Boolean(bool),
    List(List),
    Object(~Object),
    Null,
}

pub type List = ~[Json];
pub type Object = TreeMap<~str, Json>;

#[deriving(Eq)]
/// If an error occurs while parsing some JSON, this is the structure which is
/// returned
pub struct Error {
    /// The line number at which the error occurred
    priv line: uint,
    /// The column number at which the error occurred
    priv col: uint,
    /// A message describing the type of the error
    priv msg: ~str,
}

fn io_error_to_error(io: io::IoError) -> Error {
    Error {
        line: 0,
        col: 0,
        msg: format!("io error: {}", io)
    }
}

fn escape_str(s: &str) -> ~str {
    let mut escaped = ~"\"";
    for c in s.chars() {
        match c {
          '"' => escaped.push_str("\\\""),
          '\\' => escaped.push_str("\\\\"),
          '\x08' => escaped.push_str("\\b"),
          '\x0c' => escaped.push_str("\\f"),
          '\n' => escaped.push_str("\\n"),
          '\r' => escaped.push_str("\\r"),
          '\t' => escaped.push_str("\\t"),
          _ => escaped.push_char(c),
        }
    };

    escaped.push_char('"');

    escaped
}

fn spaces(n: uint) -> ~str {
    let mut ss = ~"";
    for _ in range(0, n) { ss.push_str(" "); }
    return ss;
}

/// A structure for implementing serialization to JSON.
pub struct Encoder<'a> {
    priv wr: &'a mut io::Writer,
    priv error: io::IoResult<()>,
}

impl<'a> Encoder<'a> {
    /// Creates a new JSON encoder whose output will be written to the writer
    /// specified.
    pub fn new<'a>(wr: &'a mut io::Writer) -> Encoder<'a> {
        Encoder { wr: wr, error: Ok(()) }
    }

    /// Encode the specified struct into a json [u8]
    pub fn buffer_encode<T:serialize::Encodable<Encoder<'a>>>(to_encode_object: &T) -> ~[u8]  {
       //Serialize the object in a string using a writer
        let mut m = MemWriter::new();
        {
            let mut encoder = Encoder::new(&mut m as &mut io::Writer);
            to_encode_object.encode(&mut encoder);
        }
        m.unwrap()
    }

    /// Encode the specified struct into a json str
    pub fn str_encode<T:serialize::Encodable<Encoder<'a>>>(to_encode_object: &T) -> ~str  {
        let buff:~[u8] = Encoder::buffer_encode(to_encode_object);
        str::from_utf8_owned(buff).unwrap()
    }
}

impl<'a> serialize::Encoder for Encoder<'a> {
    fn emit_nil(&mut self) { if_ok!(write!(self.wr, "null")) }

    fn emit_uint(&mut self, v: uint) { self.emit_f64(v as f64); }
    fn emit_u64(&mut self, v: u64) { self.emit_f64(v as f64); }
    fn emit_u32(&mut self, v: u32) { self.emit_f64(v as f64); }
    fn emit_u16(&mut self, v: u16) { self.emit_f64(v as f64); }
    fn emit_u8(&mut self, v: u8)   { self.emit_f64(v as f64); }

    fn emit_int(&mut self, v: int) { self.emit_f64(v as f64); }
    fn emit_i64(&mut self, v: i64) { self.emit_f64(v as f64); }
    fn emit_i32(&mut self, v: i32) { self.emit_f64(v as f64); }
    fn emit_i16(&mut self, v: i16) { self.emit_f64(v as f64); }
    fn emit_i8(&mut self, v: i8)   { self.emit_f64(v as f64); }

    fn emit_bool(&mut self, v: bool) {
        if v {
            if_ok!(write!(self.wr, "true"));
        } else {
            if_ok!(write!(self.wr, "false"));
        }
    }

    fn emit_f64(&mut self, v: f64) {
        if_ok!(write!(self.wr, "{}", f64::to_str_digits(v, 6u)))
    }
    fn emit_f32(&mut self, v: f32) { self.emit_f64(v as f64); }

    fn emit_char(&mut self, v: char) { self.emit_str(str::from_char(v)) }
    fn emit_str(&mut self, v: &str) {
        if_ok!(write!(self.wr, "{}", escape_str(v)))
    }

    fn emit_enum(&mut self, _name: &str, f: |&mut Encoder<'a>|) { f(self) }

    fn emit_enum_variant(&mut self,
                         name: &str,
                         _id: uint,
                         cnt: uint,
                         f: |&mut Encoder<'a>|) {
        // enums are encoded as strings or objects
        // Bunny => "Bunny"
        // Kangaroo(34,"William") => {"variant": "Kangaroo", "fields": [34,"William"]}
        if cnt == 0 {
            if_ok!(write!(self.wr, "{}", escape_str(name)));
        } else {
            if_ok!(write!(self.wr, "\\{\"variant\":"));
            if_ok!(write!(self.wr, "{}", escape_str(name)));
            if_ok!(write!(self.wr, ",\"fields\":["));
            f(self);
            if_ok!(write!(self.wr, "]\\}"));
        }
    }

    fn emit_enum_variant_arg(&mut self, idx: uint, f: |&mut Encoder<'a>|) {
        if idx != 0 {
            if_ok!(write!(self.wr, ","));
        }
        f(self);
    }

    fn emit_enum_struct_variant(&mut self,
                                name: &str,
                                id: uint,
                                cnt: uint,
                                f: |&mut Encoder<'a>|) {
        self.emit_enum_variant(name, id, cnt, f)
    }

    fn emit_enum_struct_variant_field(&mut self,
                                      _: &str,
                                      idx: uint,
                                      f: |&mut Encoder<'a>|) {
        self.emit_enum_variant_arg(idx, f)
    }

    fn emit_struct(&mut self, _: &str, _: uint, f: |&mut Encoder<'a>|) {
        if_ok!(write!(self.wr, r"\{"));
        f(self);
        if_ok!(write!(self.wr, r"\}"));
    }

    fn emit_struct_field(&mut self,
                         name: &str,
                         idx: uint,
                         f: |&mut Encoder<'a>|) {
        if idx != 0 { if_ok!(write!(self.wr, ",")) }
        if_ok!(write!(self.wr, "{}:", escape_str(name)));
        f(self);
    }

    fn emit_tuple(&mut self, len: uint, f: |&mut Encoder<'a>|) {
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg(&mut self, idx: uint, f: |&mut Encoder<'a>|) {
        self.emit_seq_elt(idx, f)
    }

    fn emit_tuple_struct(&mut self,
                         _name: &str,
                         len: uint,
                         f: |&mut Encoder<'a>|) {
        self.emit_seq(len, f)
    }
    fn emit_tuple_struct_arg(&mut self, idx: uint, f: |&mut Encoder<'a>|) {
        self.emit_seq_elt(idx, f)
    }

    fn emit_option(&mut self, f: |&mut Encoder<'a>|) { f(self); }
    fn emit_option_none(&mut self) { self.emit_nil(); }
    fn emit_option_some(&mut self, f: |&mut Encoder<'a>|) { f(self); }

    fn emit_seq(&mut self, _len: uint, f: |&mut Encoder<'a>|) {
        if_ok!(write!(self.wr, "["));
        f(self);
        if_ok!(write!(self.wr, "]"));
    }

    fn emit_seq_elt(&mut self, idx: uint, f: |&mut Encoder<'a>|) {
        if idx != 0 {
            if_ok!(write!(self.wr, ","));
        }
        f(self)
    }

    fn emit_map(&mut self, _len: uint, f: |&mut Encoder<'a>|) {
        if_ok!(write!(self.wr, r"\{"));
        f(self);
        if_ok!(write!(self.wr, r"\}"));
    }

    fn emit_map_elt_key(&mut self, idx: uint, f: |&mut Encoder<'a>|) {
        if idx != 0 { if_ok!(write!(self.wr, ",")) }
        f(self)
    }

    fn emit_map_elt_val(&mut self, _idx: uint, f: |&mut Encoder<'a>|) {
        if_ok!(write!(self.wr, ":"));
        f(self)
    }
}

/// Another encoder for JSON, but prints out human-readable JSON instead of
/// compact data
pub struct PrettyEncoder<'a> {
    priv wr: &'a mut io::Writer,
    priv indent: uint,
    priv error: io::IoResult<()>,
}

impl<'a> PrettyEncoder<'a> {
    /// Creates a new encoder whose output will be written to the specified writer
    pub fn new<'a>(wr: &'a mut io::Writer) -> PrettyEncoder<'a> {
        PrettyEncoder {
            wr: wr,
            indent: 0,
            error: Ok(())
        }
    }
}

impl<'a> serialize::Encoder for PrettyEncoder<'a> {
    fn emit_nil(&mut self) { if_ok!(write!(self.wr, "null")); }

    fn emit_uint(&mut self, v: uint) { self.emit_f64(v as f64); }
    fn emit_u64(&mut self, v: u64) { self.emit_f64(v as f64); }
    fn emit_u32(&mut self, v: u32) { self.emit_f64(v as f64); }
    fn emit_u16(&mut self, v: u16) { self.emit_f64(v as f64); }
    fn emit_u8(&mut self, v: u8)   { self.emit_f64(v as f64); }

    fn emit_int(&mut self, v: int) { self.emit_f64(v as f64); }
    fn emit_i64(&mut self, v: i64) { self.emit_f64(v as f64); }
    fn emit_i32(&mut self, v: i32) { self.emit_f64(v as f64); }
    fn emit_i16(&mut self, v: i16) { self.emit_f64(v as f64); }
    fn emit_i8(&mut self, v: i8)   { self.emit_f64(v as f64); }

    fn emit_bool(&mut self, v: bool) {
        if v {
            if_ok!(write!(self.wr, "true"));
        } else {
            if_ok!(write!(self.wr, "false"));
        }
    }

    fn emit_f64(&mut self, v: f64) {
        if_ok!(write!(self.wr, "{}", f64::to_str_digits(v, 6u)));
    }
    fn emit_f32(&mut self, v: f32) { self.emit_f64(v as f64); }

    fn emit_char(&mut self, v: char) { self.emit_str(str::from_char(v)) }
    fn emit_str(&mut self, v: &str) {
        if_ok!(write!(self.wr, "{}", escape_str(v)));
    }

    fn emit_enum(&mut self, _name: &str, f: |&mut PrettyEncoder<'a>|) {
        f(self)
    }

    fn emit_enum_variant(&mut self,
                         name: &str,
                         _: uint,
                         cnt: uint,
                         f: |&mut PrettyEncoder<'a>|) {
        if cnt == 0 {
            if_ok!(write!(self.wr, "{}", escape_str(name)));
        } else {
            self.indent += 2;
            if_ok!(write!(self.wr, "[\n{}{},\n", spaces(self.indent),
                          escape_str(name)));
            f(self);
            self.indent -= 2;
            if_ok!(write!(self.wr, "\n{}]", spaces(self.indent)));
        }
    }

    fn emit_enum_variant_arg(&mut self,
                             idx: uint,
                             f: |&mut PrettyEncoder<'a>|) {
        if idx != 0 {
            if_ok!(write!(self.wr, ",\n"));
        }
        if_ok!(write!(self.wr, "{}", spaces(self.indent)));
        f(self)
    }

    fn emit_enum_struct_variant(&mut self,
                                name: &str,
                                id: uint,
                                cnt: uint,
                                f: |&mut PrettyEncoder<'a>|) {
        self.emit_enum_variant(name, id, cnt, f)
    }

    fn emit_enum_struct_variant_field(&mut self,
                                      _: &str,
                                      idx: uint,
                                      f: |&mut PrettyEncoder<'a>|) {
        self.emit_enum_variant_arg(idx, f)
    }


    fn emit_struct(&mut self,
                   _: &str,
                   len: uint,
                   f: |&mut PrettyEncoder<'a>|) {
        if len == 0 {
            if_ok!(write!(self.wr, "\\{\\}"));
        } else {
            if_ok!(write!(self.wr, "\\{"));
            self.indent += 2;
            f(self);
            self.indent -= 2;
            if_ok!(write!(self.wr, "\n{}\\}", spaces(self.indent)));
        }
    }

    fn emit_struct_field(&mut self,
                         name: &str,
                         idx: uint,
                         f: |&mut PrettyEncoder<'a>|) {
        if idx == 0 {
            if_ok!(write!(self.wr, "\n"));
        } else {
            if_ok!(write!(self.wr, ",\n"));
        }
        if_ok!(write!(self.wr, "{}{}: ", spaces(self.indent), escape_str(name)));
        f(self);
    }

    fn emit_tuple(&mut self, len: uint, f: |&mut PrettyEncoder<'a>|) {
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg(&mut self, idx: uint, f: |&mut PrettyEncoder<'a>|) {
        self.emit_seq_elt(idx, f)
    }

    fn emit_tuple_struct(&mut self,
                         _: &str,
                         len: uint,
                         f: |&mut PrettyEncoder<'a>|) {
        self.emit_seq(len, f)
    }
    fn emit_tuple_struct_arg(&mut self,
                             idx: uint,
                             f: |&mut PrettyEncoder<'a>|) {
        self.emit_seq_elt(idx, f)
    }

    fn emit_option(&mut self, f: |&mut PrettyEncoder<'a>|) { f(self); }
    fn emit_option_none(&mut self) { self.emit_nil(); }
    fn emit_option_some(&mut self, f: |&mut PrettyEncoder<'a>|) { f(self); }

    fn emit_seq(&mut self, len: uint, f: |&mut PrettyEncoder<'a>|) {
        if len == 0 {
            if_ok!(write!(self.wr, "[]"));
        } else {
            if_ok!(write!(self.wr, "["));
            self.indent += 2;
            f(self);
            self.indent -= 2;
            if_ok!(write!(self.wr, "\n{}]", spaces(self.indent)));
        }
    }

    fn emit_seq_elt(&mut self, idx: uint, f: |&mut PrettyEncoder<'a>|) {
        if idx == 0 {
            if_ok!(write!(self.wr, "\n"));
        } else {
            if_ok!(write!(self.wr, ",\n"));
        }
        if_ok!(write!(self.wr, "{}", spaces(self.indent)));
        f(self)
    }

    fn emit_map(&mut self, len: uint, f: |&mut PrettyEncoder<'a>|) {
        if len == 0 {
            if_ok!(write!(self.wr, "\\{\\}"));
        } else {
            if_ok!(write!(self.wr, "\\{"));
            self.indent += 2;
            f(self);
            self.indent -= 2;
            if_ok!(write!(self.wr, "\n{}\\}", spaces(self.indent)));
        }
    }

    fn emit_map_elt_key(&mut self, idx: uint, f: |&mut PrettyEncoder<'a>|) {
        if idx == 0 {
            if_ok!(write!(self.wr, "\n"));
        } else {
            if_ok!(write!(self.wr, ",\n"));
        }
        if_ok!(write!(self.wr, "{}", spaces(self.indent)));
        f(self);
    }

    fn emit_map_elt_val(&mut self, _idx: uint, f: |&mut PrettyEncoder<'a>|) {
        if_ok!(write!(self.wr, ": "));
        f(self);
    }
}

impl<E: serialize::Encoder> serialize::Encodable<E> for Json {
    fn encode(&self, e: &mut E) {
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

impl Json {
    /// Encodes a json value into a io::writer.  Uses a single line.
    pub fn to_writer(&self, wr: &mut io::Writer) -> io::IoResult<()> {
        let mut encoder = Encoder::new(wr);
        self.encode(&mut encoder);
        encoder.error
    }

    /// Encodes a json value into a io::writer.
    /// Pretty-prints in a more readable format.
    pub fn to_pretty_writer(&self, wr: &mut io::Writer) -> io::IoResult<()> {
        let mut encoder = PrettyEncoder::new(wr);
        self.encode(&mut encoder);
        encoder.error
    }

    /// Encodes a json value into a string
    pub fn to_pretty_str(&self) -> ~str {
        let mut s = MemWriter::new();
        self.to_pretty_writer(&mut s as &mut io::Writer).unwrap();
        str::from_utf8_owned(s.unwrap()).unwrap()
    }
}

pub struct Parser<T> {
    priv rdr: T,
    priv ch: char,
    priv line: uint,
    priv col: uint,
}

impl<T: Iterator<char>> Parser<T> {
    /// Decode a json value from an Iterator<char>
    pub fn new(rdr: T) -> Parser<T> {
        let mut p = Parser {
            rdr: rdr,
            ch: '\x00',
            line: 1,
            col: 0,
        };
        p.bump();
        p
    }
}

impl<T: Iterator<char>> Parser<T> {
    pub fn parse(&mut self) -> Result<Json, Error> {
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

impl<T : Iterator<char>> Parser<T> {
    // FIXME: #8971: unsound
    fn eof(&self) -> bool { self.ch == unsafe { transmute(-1u32) } }

    fn bump(&mut self) {
        match self.rdr.next() {
            Some(ch) => self.ch = ch,
            None() => self.ch = unsafe { transmute(-1u32) }, // FIXME: #8971: unsound
        }

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
        Err(Error { line: self.line, col: self.col, msg: msg })
    }

    fn parse_value(&mut self) -> Result<Json, Error> {
        self.parse_whitespace();

        if self.eof() { return self.error(~"EOF while parsing value"); }

        match self.ch {
          'n' => self.parse_ident("ull", Null),
          't' => self.parse_ident("rue", Boolean(true)),
          'f' => self.parse_ident("alse", Boolean(false)),
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
        while self.ch == ' ' ||
              self.ch == '\n' ||
              self.ch == '\t' ||
              self.ch == '\r' { self.bump(); }
    }

    fn parse_ident(&mut self, ident: &str, value: Json) -> Result<Json, Error> {
        if ident.chars().all(|c| c == self.next_char()) {
            self.bump();
            Ok(value)
        } else {
            self.error(~"invalid syntax")
        }
    }

    fn parse_number(&mut self) -> Result<Json, Error> {
        let mut neg = 1.0;

        if self.ch == '-' {
            self.bump();
            neg = -1.0;
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

    fn parse_integer(&mut self) -> Result<f64, Error> {
        let mut res = 0.0;

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
                    res *= 10.0;
                    res += ((self.ch as int) - ('0' as int)) as f64;

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

    fn parse_decimal(&mut self, res: f64) -> Result<f64, Error> {
        self.bump();

        // Make sure a digit follows the decimal place.
        match self.ch {
          '0' .. '9' => (),
          _ => return self.error(~"invalid number")
        }

        let mut res = res;
        let mut dec = 1.0;
        while !self.eof() {
            match self.ch {
              '0' .. '9' => {
                dec /= 10.0;
                res += (((self.ch as int) - ('0' as int)) as f64) * dec;

                self.bump();
              }
              _ => break
            }
        }

        Ok(res)
    }

    fn parse_exponent(&mut self, mut res: f64) -> Result<f64, Error> {
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

        let exp: f64 = num::pow(10u as f64, exp);
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

        loop {
            self.bump();
            if self.eof() {
                return self.error(~"EOF while parsing string");
            }

            if escape {
                match self.ch {
                  '"' => res.push_char('"'),
                  '\\' => res.push_char('\\'),
                  '/' => res.push_char('/'),
                  'b' => res.push_char('\x08'),
                  'f' => res.push_char('\x0c'),
                  'n' => res.push_char('\n'),
                  'r' => res.push_char('\r'),
                  't' => res.push_char('\t'),
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

                      res.push_char(char::from_u32(n as u32).unwrap());
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
                res.push_char(self.ch);
            }
        }
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

        let mut values = ~TreeMap::new();

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

/// Decodes a json value from an `&mut io::Reader`
pub fn from_reader(rdr: &mut io::Reader) -> Result<Json, Error> {
    let contents = match rdr.read_to_end() {
        Ok(c) => c,
        Err(e) => return Err(io_error_to_error(e))
    };
    let s = match str::from_utf8_owned(contents) {
        Some(s) => s,
        None => return Err(Error { line: 0, col: 0, msg: ~"contents not utf-8" })
    };
    let mut parser = Parser::new(s.chars());
    parser.parse()
}

/// Decodes a json value from a string
pub fn from_str(s: &str) -> Result<Json, Error> {
    let mut parser = Parser::new(s.chars());
    parser.parse()
}

/// A structure to decode JSON to values in rust.
pub struct Decoder {
    priv stack: ~[Json],
}

impl Decoder {
    /// Creates a new decoder instance for decoding the specified JSON value.
    pub fn new(json: Json) -> Decoder {
        Decoder {
            stack: ~[json]
        }
    }
}

impl Decoder {
    fn err(&self, msg: &str) -> ! {
        fail!("JSON decode error: {}", msg);
    }
    fn missing_field(&self, field: &str, object: ~Object) -> ! {
        self.err(format!("missing required '{}' field in object: {}",
                         field, Object(object).to_str()))
    }
    fn expected(&self, expected: &str, found: &Json) -> ! {
        let found_s = match *found {
            Null => "null",
            List(..) => "list",
            Object(..) => "object",
            Number(..) => "number",
            String(..) => "string",
            Boolean(..) => "boolean"
        };
        self.err(format!("expected {expct} but found {fnd}: {val}",
                         expct=expected, fnd=found_s, val=found.to_str()))
    }
}

impl serialize::Decoder for Decoder {
    fn read_nil(&mut self) -> () {
        debug!("read_nil");
        match self.stack.pop().unwrap() {
            Null => (),
            value => self.expected("null", &value)
        }
    }

    fn read_u64(&mut self)  -> u64  { self.read_f64() as u64 }
    fn read_u32(&mut self)  -> u32  { self.read_f64() as u32 }
    fn read_u16(&mut self)  -> u16  { self.read_f64() as u16 }
    fn read_u8 (&mut self)  -> u8   { self.read_f64() as u8 }
    fn read_uint(&mut self) -> uint { self.read_f64() as uint }

    fn read_i64(&mut self) -> i64 { self.read_f64() as i64 }
    fn read_i32(&mut self) -> i32 { self.read_f64() as i32 }
    fn read_i16(&mut self) -> i16 { self.read_f64() as i16 }
    fn read_i8 (&mut self) -> i8  { self.read_f64() as i8 }
    fn read_int(&mut self) -> int { self.read_f64() as int }

    fn read_bool(&mut self) -> bool {
        debug!("read_bool");
        match self.stack.pop().unwrap() {
            Boolean(b) => b,
            value => self.expected("boolean", &value)
        }
    }

    fn read_f64(&mut self) -> f64 {
        debug!("read_f64");
        match self.stack.pop().unwrap() {
            Number(f) => f,
            value => self.expected("number", &value)
        }
    }
    fn read_f32(&mut self) -> f32 { self.read_f64() as f32 }
    fn read_f32(&mut self) -> f32 { self.read_f64() as f32 }

    fn read_char(&mut self) -> char {
        let s = self.read_str();
        {
            let mut it = s.chars();
            match (it.next(), it.next()) {
                // exactly one character
                (Some(c), None) => return c,
                _ => ()
            }
        }
        self.expected("single character string", &String(s))
    }

    fn read_str(&mut self) -> ~str {
        debug!("read_str");
        match self.stack.pop().unwrap() {
            String(s) => s,
            value => self.expected("string", &value)
        }
    }

    fn read_enum<T>(&mut self, name: &str, f: |&mut Decoder| -> T) -> T {
        debug!("read_enum({})", name);
        f(self)
    }

    fn read_enum_variant<T>(&mut self,
                            names: &[&str],
                            f: |&mut Decoder, uint| -> T)
                            -> T {
        debug!("read_enum_variant(names={:?})", names);
        let name = match self.stack.pop().unwrap() {
            String(s) => s,
            Object(mut o) => {
                let n = match o.pop(&~"variant") {
                    Some(String(s)) => s,
                    Some(val) => self.expected("string", &val),
                    None => self.missing_field("variant", o)
                };
                match o.pop(&~"fields") {
                    Some(List(l)) => {
                        for field in l.move_rev_iter() {
                            self.stack.push(field.clone());
                        }
                    },
                    Some(val) => self.expected("list", &val),
                    None => {
                        // re-insert the variant field so we're
                        // printing the "whole" struct in the error
                        // message... ick.
                        o.insert(~"variant", String(n));
                        self.missing_field("fields", o);
                    }
                }
                n
            }
            json => self.expected("string or object", &json)
        };
        let idx = match names.iter().position(|n| str::eq_slice(*n, name)) {
            Some(idx) => idx,
            None => self.err(format!("unknown variant name: {}", name))
        };
        f(self, idx)
    }

    fn read_enum_variant_arg<T>(&mut self, idx: uint, f: |&mut Decoder| -> T)
                                -> T {
        debug!("read_enum_variant_arg(idx={})", idx);
        f(self)
    }

    fn read_enum_struct_variant<T>(&mut self,
                                   names: &[&str],
                                   f: |&mut Decoder, uint| -> T)
                                   -> T {
        debug!("read_enum_struct_variant(names={:?})", names);
        self.read_enum_variant(names, f)
    }


    fn read_enum_struct_variant_field<T>(&mut self,
                                         name: &str,
                                         idx: uint,
                                         f: |&mut Decoder| -> T)
                                         -> T {
        debug!("read_enum_struct_variant_field(name={}, idx={})", name, idx);
        self.read_enum_variant_arg(idx, f)
    }

    fn read_struct<T>(&mut self,
                      name: &str,
                      len: uint,
                      f: |&mut Decoder| -> T)
                      -> T {
        debug!("read_struct(name={}, len={})", name, len);
        let value = f(self);
        self.stack.pop().unwrap();
        value
    }

    fn read_struct_field<T>(&mut self,
                            name: &str,
                            idx: uint,
                            f: |&mut Decoder| -> T)
                            -> T {
        debug!("read_struct_field(name={}, idx={})", name, idx);
        match self.stack.pop().unwrap() {
            Object(mut obj) => {
                let value = match obj.pop(&name.to_owned()) {
                    None => self.missing_field(name, obj),
                    Some(json) => {
                        self.stack.push(json);
                        f(self)
                    }
                };
                self.stack.push(Object(obj));
                value
            }
            value => self.expected("object", &value)
        }
    }

    fn read_tuple<T>(&mut self, f: |&mut Decoder, uint| -> T) -> T {
        debug!("read_tuple()");
        self.read_seq(f)
    }

    fn read_tuple_arg<T>(&mut self, idx: uint, f: |&mut Decoder| -> T) -> T {
        debug!("read_tuple_arg(idx={})", idx);
        self.read_seq_elt(idx, f)
    }

    fn read_tuple_struct<T>(&mut self,
                            name: &str,
                            f: |&mut Decoder, uint| -> T)
                            -> T {
        debug!("read_tuple_struct(name={})", name);
        self.read_tuple(f)
    }

    fn read_tuple_struct_arg<T>(&mut self,
                                idx: uint,
                                f: |&mut Decoder| -> T)
                                -> T {
        debug!("read_tuple_struct_arg(idx={})", idx);
        self.read_tuple_arg(idx, f)
    }

    fn read_option<T>(&mut self, f: |&mut Decoder, bool| -> T) -> T {
        match self.stack.pop().unwrap() {
            Null => f(self, false),
            value => { self.stack.push(value); f(self, true) }
        }
    }

    fn read_seq<T>(&mut self, f: |&mut Decoder, uint| -> T) -> T {
        debug!("read_seq()");
        let len = match self.stack.pop().unwrap() {
            List(list) => {
                let len = list.len();
                for v in list.move_rev_iter() {
                    self.stack.push(v);
                }
                len
            }
            value => self.expected("list", &value)
        };
        f(self, len)
    }

    fn read_seq_elt<T>(&mut self, idx: uint, f: |&mut Decoder| -> T) -> T {
        debug!("read_seq_elt(idx={})", idx);
        f(self)
    }

    fn read_map<T>(&mut self, f: |&mut Decoder, uint| -> T) -> T {
        debug!("read_map()");
        let len = match self.stack.pop().unwrap() {
            Object(obj) => {
                let len = obj.len();
                for (key, value) in obj.move_iter() {
                    self.stack.push(value);
                    self.stack.push(String(key));
                }
                len
            }
            value => self.expected("object", &value)
        };
        f(self, len)
    }

    fn read_map_elt_key<T>(&mut self, idx: uint, f: |&mut Decoder| -> T)
                           -> T {
        debug!("read_map_elt_key(idx={})", idx);
        f(self)
    }

    fn read_map_elt_val<T>(&mut self, idx: uint, f: |&mut Decoder| -> T)
                           -> T {
        debug!("read_map_elt_val(idx={})", idx);
        f(self)
    }
}

/// Test if two json values are less than one another
impl Ord for Json {
    fn lt(&self, other: &Json) -> bool {
        match *self {
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
                    Object(ref d1) => d0 < d1,
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
}

/// A trait for converting values to JSON
pub trait ToJson {
    /// Converts the value of `self` to an instance of JSON
    fn to_json(&self) -> Json;
}

impl ToJson for Json {
    fn to_json(&self) -> Json { (*self).clone() }
}

impl ToJson for int {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for i8 {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for i16 {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for i32 {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for i64 {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for uint {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for u8 {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for u16 {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for u32 {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for u64 {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for f32 {
    fn to_json(&self) -> Json { Number(*self as f64) }
}

impl ToJson for f64 {
    fn to_json(&self) -> Json { Number(*self) }
}

impl ToJson for () {
    fn to_json(&self) -> Json { Null }
}

impl ToJson for bool {
    fn to_json(&self) -> Json { Boolean(*self) }
}

impl ToJson for ~str {
    fn to_json(&self) -> Json { String((*self).clone()) }
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

impl<A:ToJson> ToJson for TreeMap<~str, A> {
    fn to_json(&self) -> Json {
        let mut d = TreeMap::new();
        for (key, value) in self.iter() {
            d.insert((*key).clone(), value.to_json());
        }
        Object(~d)
    }
}

impl<A:ToJson> ToJson for HashMap<~str, A> {
    fn to_json(&self) -> Json {
        let mut d = TreeMap::new();
        for (key, value) in self.iter() {
            d.insert((*key).clone(), value.to_json());
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
    /// Encodes a json value into a string
    fn to_str(&self) -> ~str {
        let mut s = MemWriter::new();
        self.to_writer(&mut s as &mut io::Writer).unwrap();
        str::from_utf8_owned(s.unwrap()).unwrap()
    }
}

impl to_str::ToStr for Error {
    fn to_str(&self) -> ~str {
        format!("{}:{}: {}", self.line, self.col, self.msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io;
    use serialize::{Encodable, Decodable};
    use treemap::TreeMap;

    #[deriving(Eq, Encodable, Decodable)]
    enum Animal {
        Dog,
        Frog(~str, int)
    }

    #[deriving(Eq, Encodable, Decodable)]
    struct Inner {
        a: (),
        b: uint,
        c: ~[~str],
    }

    #[deriving(Eq, Encodable, Decodable)]
    struct Outer {
        inner: ~[Inner],
    }

    fn mk_object(items: &[(~str, Json)]) -> Json {
        let mut d = ~TreeMap::new();

        for item in items.iter() {
            match *item {
                (ref key, ref value) => { d.insert((*key).clone(), (*value).clone()); },
            }
        };

        Object(d)
    }

    #[test]
    fn test_write_null() {
        assert_eq!(Null.to_str(), ~"null");
        assert_eq!(Null.to_pretty_str(), ~"null");
    }


    #[test]
    fn test_write_number() {
        assert_eq!(Number(3.0).to_str(), ~"3");
        assert_eq!(Number(3.0).to_pretty_str(), ~"3");

        assert_eq!(Number(3.1).to_str(), ~"3.1");
        assert_eq!(Number(3.1).to_pretty_str(), ~"3.1");

        assert_eq!(Number(-1.5).to_str(), ~"-1.5");
        assert_eq!(Number(-1.5).to_pretty_str(), ~"-1.5");

        assert_eq!(Number(0.5).to_str(), ~"0.5");
        assert_eq!(Number(0.5).to_pretty_str(), ~"0.5");
    }

    #[test]
    fn test_write_str() {
        assert_eq!(String(~"").to_str(), ~"\"\"");
        assert_eq!(String(~"").to_pretty_str(), ~"\"\"");

        assert_eq!(String(~"foo").to_str(), ~"\"foo\"");
        assert_eq!(String(~"foo").to_pretty_str(), ~"\"foo\"");
    }

    #[test]
    fn test_write_bool() {
        assert_eq!(Boolean(true).to_str(), ~"true");
        assert_eq!(Boolean(true).to_pretty_str(), ~"true");

        assert_eq!(Boolean(false).to_str(), ~"false");
        assert_eq!(Boolean(false).to_pretty_str(), ~"false");
    }

    #[test]
    fn test_write_list() {
        assert_eq!(List(~[]).to_str(), ~"[]");
        assert_eq!(List(~[]).to_pretty_str(), ~"[]");

        assert_eq!(List(~[Boolean(true)]).to_str(), ~"[true]");
        assert_eq!(
            List(~[Boolean(true)]).to_pretty_str(),
            ~"\
            [\n  \
                true\n\
            ]"
        );

        let longTestList = List(~[
            Boolean(false),
            Null,
            List(~[String(~"foo\nbar"), Number(3.5)])]);

        assert_eq!(longTestList.to_str(),
            ~"[false,null,[\"foo\\nbar\",3.5]]");
        assert_eq!(
            longTestList.to_pretty_str(),
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
        assert_eq!(mk_object([]).to_str(), ~"{}");
        assert_eq!(mk_object([]).to_pretty_str(), ~"{}");

        assert_eq!(
            mk_object([(~"a", Boolean(true))]).to_str(),
            ~"{\"a\":true}"
        );
        assert_eq!(
            mk_object([(~"a", Boolean(true))]).to_pretty_str(),
            ~"\
            {\n  \
                \"a\": true\n\
            }"
        );

        let complexObj = mk_object([
                (~"b", List(~[
                    mk_object([(~"c", String(~"\x0c\r"))]),
                    mk_object([(~"d", String(~""))])
                ]))
            ]);

        assert_eq!(
            complexObj.to_str(),
            ~"{\
                \"b\":[\
                    {\"c\":\"\\f\\r\"},\
                    {\"d\":\"\"}\
                ]\
            }"
        );
        assert_eq!(
            complexObj.to_pretty_str(),
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

        let a = mk_object([
            (~"a", Boolean(true)),
            (~"b", List(~[
                mk_object([(~"c", String(~"\x0c\r"))]),
                mk_object([(~"d", String(~""))])
            ]))
        ]);

        // We can't compare the strings directly because the object fields be
        // printed in a different order.
        assert_eq!(a.clone(), from_str(a.to_str()).unwrap());
        assert_eq!(a.clone(), from_str(a.to_pretty_str()).unwrap());
    }

    fn with_str_writer(f: |&mut io::Writer|) -> ~str {
        use std::io::MemWriter;
        use std::str;

        let mut m = MemWriter::new();
        f(&mut m as &mut io::Writer);
        str::from_utf8_owned(m.unwrap()).unwrap()
    }

    #[test]
    fn test_write_enum() {
        let animal = Dog;
        assert_eq!(
            with_str_writer(|wr| {
                let mut encoder = Encoder::new(wr);
                animal.encode(&mut encoder);
            }),
            ~"\"Dog\""
        );
        assert_eq!(
            with_str_writer(|wr| {
                let mut encoder = PrettyEncoder::new(wr);
                animal.encode(&mut encoder);
            }),
            ~"\"Dog\""
        );

        let animal = Frog(~"Henry", 349);
        assert_eq!(
            with_str_writer(|wr| {
                let mut encoder = Encoder::new(wr);
                animal.encode(&mut encoder);
            }),
            ~"{\"variant\":\"Frog\",\"fields\":[\"Henry\",349]}"
        );
        assert_eq!(
            with_str_writer(|wr| {
                let mut encoder = PrettyEncoder::new(wr);
                animal.encode(&mut encoder);
            }),
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
        let s = with_str_writer(|wr| {
            let mut encoder = Encoder::new(wr);
            value.encode(&mut encoder);
        });
        assert_eq!(s, ~"\"jodhpurs\"");

        let value = Some(~"jodhpurs");
        let s = with_str_writer(|wr| {
            let mut encoder = PrettyEncoder::new(wr);
            value.encode(&mut encoder);
        });
        assert_eq!(s, ~"\"jodhpurs\"");
    }

    #[test]
    fn test_write_none() {
        let value: Option<~str> = None;
        let s = with_str_writer(|wr| {
            let mut encoder = Encoder::new(wr);
            value.encode(&mut encoder);
        });
        assert_eq!(s, ~"null");

        let s = with_str_writer(|wr| {
            let mut encoder = Encoder::new(wr);
            value.encode(&mut encoder);
        });
        assert_eq!(s, ~"null");
    }

    #[test]
    fn test_trailing_characters() {
        assert_eq!(from_str("nulla"),
            Err(Error {line: 1u, col: 5u, msg: ~"trailing characters"}));
        assert_eq!(from_str("truea"),
            Err(Error {line: 1u, col: 5u, msg: ~"trailing characters"}));
        assert_eq!(from_str("falsea"),
            Err(Error {line: 1u, col: 6u, msg: ~"trailing characters"}));
        assert_eq!(from_str("1a"),
            Err(Error {line: 1u, col: 2u, msg: ~"trailing characters"}));
        assert_eq!(from_str("[]a"),
            Err(Error {line: 1u, col: 3u, msg: ~"trailing characters"}));
        assert_eq!(from_str("{}a"),
            Err(Error {line: 1u, col: 3u, msg: ~"trailing characters"}));
    }

    #[test]
    fn test_read_identifiers() {
        assert_eq!(from_str("n"),
            Err(Error {line: 1u, col: 2u, msg: ~"invalid syntax"}));
        assert_eq!(from_str("nul"),
            Err(Error {line: 1u, col: 4u, msg: ~"invalid syntax"}));

        assert_eq!(from_str("t"),
            Err(Error {line: 1u, col: 2u, msg: ~"invalid syntax"}));
        assert_eq!(from_str("truz"),
            Err(Error {line: 1u, col: 4u, msg: ~"invalid syntax"}));

        assert_eq!(from_str("f"),
            Err(Error {line: 1u, col: 2u, msg: ~"invalid syntax"}));
        assert_eq!(from_str("faz"),
            Err(Error {line: 1u, col: 3u, msg: ~"invalid syntax"}));

        assert_eq!(from_str("null"), Ok(Null));
        assert_eq!(from_str("true"), Ok(Boolean(true)));
        assert_eq!(from_str("false"), Ok(Boolean(false)));
        assert_eq!(from_str(" null "), Ok(Null));
        assert_eq!(from_str(" true "), Ok(Boolean(true)));
        assert_eq!(from_str(" false "), Ok(Boolean(false)));
    }

    #[test]
    fn test_decode_identifiers() {
        let mut decoder = Decoder::new(from_str("null").unwrap());
        let v: () = Decodable::decode(&mut decoder);
        assert_eq!(v, ());

        let mut decoder = Decoder::new(from_str("true").unwrap());
        let v: bool = Decodable::decode(&mut decoder);
        assert_eq!(v, true);

        let mut decoder = Decoder::new(from_str("false").unwrap());
        let v: bool = Decodable::decode(&mut decoder);
        assert_eq!(v, false);
    }

    #[test]
    fn test_read_number() {
        assert_eq!(from_str("+"),
            Err(Error {line: 1u, col: 1u, msg: ~"invalid syntax"}));
        assert_eq!(from_str("."),
            Err(Error {line: 1u, col: 1u, msg: ~"invalid syntax"}));

        assert_eq!(from_str("-"),
            Err(Error {line: 1u, col: 2u, msg: ~"invalid number"}));
        assert_eq!(from_str("00"),
            Err(Error {line: 1u, col: 2u, msg: ~"invalid number"}));
        assert_eq!(from_str("1."),
            Err(Error {line: 1u, col: 3u, msg: ~"invalid number"}));
        assert_eq!(from_str("1e"),
            Err(Error {line: 1u, col: 3u, msg: ~"invalid number"}));
        assert_eq!(from_str("1e+"),
            Err(Error {line: 1u, col: 4u, msg: ~"invalid number"}));

        assert_eq!(from_str("3"), Ok(Number(3.0)));
        assert_eq!(from_str("3.1"), Ok(Number(3.1)));
        assert_eq!(from_str("-1.2"), Ok(Number(-1.2)));
        assert_eq!(from_str("0.4"), Ok(Number(0.4)));
        assert_eq!(from_str("0.4e5"), Ok(Number(0.4e5)));
        assert_eq!(from_str("0.4e+15"), Ok(Number(0.4e15)));
        assert_eq!(from_str("0.4e-01"), Ok(Number(0.4e-01)));
        assert_eq!(from_str(" 3 "), Ok(Number(3.0)));
    }

    #[test]
    fn test_decode_numbers() {
        let mut decoder = Decoder::new(from_str("3").unwrap());
        let v: f64 = Decodable::decode(&mut decoder);
        assert_eq!(v, 3.0);

        let mut decoder = Decoder::new(from_str("3.1").unwrap());
        let v: f64 = Decodable::decode(&mut decoder);
        assert_eq!(v, 3.1);

        let mut decoder = Decoder::new(from_str("-1.2").unwrap());
        let v: f64 = Decodable::decode(&mut decoder);
        assert_eq!(v, -1.2);

        let mut decoder = Decoder::new(from_str("0.4").unwrap());
        let v: f64 = Decodable::decode(&mut decoder);
        assert_eq!(v, 0.4);

        let mut decoder = Decoder::new(from_str("0.4e5").unwrap());
        let v: f64 = Decodable::decode(&mut decoder);
        assert_eq!(v, 0.4e5);

        let mut decoder = Decoder::new(from_str("0.4e15").unwrap());
        let v: f64 = Decodable::decode(&mut decoder);
        assert_eq!(v, 0.4e15);

        let mut decoder = Decoder::new(from_str("0.4e-01").unwrap());
        let v: f64 = Decodable::decode(&mut decoder);
        assert_eq!(v, 0.4e-01);
    }

    #[test]
    fn test_read_str() {
        assert_eq!(from_str("\""),
            Err(Error {line: 1u, col: 2u, msg: ~"EOF while parsing string"
        }));
        assert_eq!(from_str("\"lol"),
            Err(Error {line: 1u, col: 5u, msg: ~"EOF while parsing string"
        }));

        assert_eq!(from_str("\"\""), Ok(String(~"")));
        assert_eq!(from_str("\"foo\""), Ok(String(~"foo")));
        assert_eq!(from_str("\"\\\"\""), Ok(String(~"\"")));
        assert_eq!(from_str("\"\\b\""), Ok(String(~"\x08")));
        assert_eq!(from_str("\"\\n\""), Ok(String(~"\n")));
        assert_eq!(from_str("\"\\r\""), Ok(String(~"\r")));
        assert_eq!(from_str("\"\\t\""), Ok(String(~"\t")));
        assert_eq!(from_str(" \"foo\" "), Ok(String(~"foo")));
        assert_eq!(from_str("\"\\u12ab\""), Ok(String(~"\u12ab")));
        assert_eq!(from_str("\"\\uAB12\""), Ok(String(~"\uAB12")));
    }

    #[test]
    fn test_decode_str() {
        let mut decoder = Decoder::new(from_str("\"\"").unwrap());
        let v: ~str = Decodable::decode(&mut decoder);
        assert_eq!(v, ~"");

        let mut decoder = Decoder::new(from_str("\"foo\"").unwrap());
        let v: ~str = Decodable::decode(&mut decoder);
        assert_eq!(v, ~"foo");

        let mut decoder = Decoder::new(from_str("\"\\\"\"").unwrap());
        let v: ~str = Decodable::decode(&mut decoder);
        assert_eq!(v, ~"\"");

        let mut decoder = Decoder::new(from_str("\"\\b\"").unwrap());
        let v: ~str = Decodable::decode(&mut decoder);
        assert_eq!(v, ~"\x08");

        let mut decoder = Decoder::new(from_str("\"\\n\"").unwrap());
        let v: ~str = Decodable::decode(&mut decoder);
        assert_eq!(v, ~"\n");

        let mut decoder = Decoder::new(from_str("\"\\r\"").unwrap());
        let v: ~str = Decodable::decode(&mut decoder);
        assert_eq!(v, ~"\r");

        let mut decoder = Decoder::new(from_str("\"\\t\"").unwrap());
        let v: ~str = Decodable::decode(&mut decoder);
        assert_eq!(v, ~"\t");

        let mut decoder = Decoder::new(from_str("\"\\u12ab\"").unwrap());
        let v: ~str = Decodable::decode(&mut decoder);
        assert_eq!(v, ~"\u12ab");

        let mut decoder = Decoder::new(from_str("\"\\uAB12\"").unwrap());
        let v: ~str = Decodable::decode(&mut decoder);
        assert_eq!(v, ~"\uAB12");
    }

    #[test]
    fn test_read_list() {
        assert_eq!(from_str("["),
            Err(Error {line: 1u, col: 2u, msg: ~"EOF while parsing value"}));
        assert_eq!(from_str("[1"),
            Err(Error {line: 1u, col: 3u, msg: ~"EOF while parsing list"}));
        assert_eq!(from_str("[1,"),
            Err(Error {line: 1u, col: 4u, msg: ~"EOF while parsing value"}));
        assert_eq!(from_str("[1,]"),
            Err(Error {line: 1u, col: 4u, msg: ~"invalid syntax"}));
        assert_eq!(from_str("[6 7]"),
            Err(Error {line: 1u, col: 4u, msg: ~"expected `,` or `]`"}));

        assert_eq!(from_str("[]"), Ok(List(~[])));
        assert_eq!(from_str("[ ]"), Ok(List(~[])));
        assert_eq!(from_str("[true]"), Ok(List(~[Boolean(true)])));
        assert_eq!(from_str("[ false ]"), Ok(List(~[Boolean(false)])));
        assert_eq!(from_str("[null]"), Ok(List(~[Null])));
        assert_eq!(from_str("[3, 1]"),
                     Ok(List(~[Number(3.0), Number(1.0)])));
        assert_eq!(from_str("\n[3, 2]\n"),
                     Ok(List(~[Number(3.0), Number(2.0)])));
        assert_eq!(from_str("[2, [4, 1]]"),
               Ok(List(~[Number(2.0), List(~[Number(4.0), Number(1.0)])])));
    }

    #[test]
    fn test_decode_list() {
        let mut decoder = Decoder::new(from_str("[]").unwrap());
        let v: ~[()] = Decodable::decode(&mut decoder);
        assert_eq!(v, ~[]);

        let mut decoder = Decoder::new(from_str("[null]").unwrap());
        let v: ~[()] = Decodable::decode(&mut decoder);
        assert_eq!(v, ~[()]);

        let mut decoder = Decoder::new(from_str("[true]").unwrap());
        let v: ~[bool] = Decodable::decode(&mut decoder);
        assert_eq!(v, ~[true]);

        let mut decoder = Decoder::new(from_str("[true]").unwrap());
        let v: ~[bool] = Decodable::decode(&mut decoder);
        assert_eq!(v, ~[true]);

        let mut decoder = Decoder::new(from_str("[3, 1]").unwrap());
        let v: ~[int] = Decodable::decode(&mut decoder);
        assert_eq!(v, ~[3, 1]);

        let mut decoder = Decoder::new(from_str("[[3], [1, 2]]").unwrap());
        let v: ~[~[uint]] = Decodable::decode(&mut decoder);
        assert_eq!(v, ~[~[3], ~[1, 2]]);
    }

    #[test]
    fn test_read_object() {
        assert_eq!(from_str("{"),
            Err(Error {
                line: 1u,
                col: 2u,
                msg: ~"EOF while parsing object"}));
        assert_eq!(from_str("{ "),
            Err(Error {
                line: 1u,
                col: 3u,
                msg: ~"EOF while parsing object"}));
        assert_eq!(from_str("{1"),
            Err(Error {
                line: 1u,
                col: 2u,
                msg: ~"key must be a string"}));
        assert_eq!(from_str("{ \"a\""),
            Err(Error {
                line: 1u,
                col: 6u,
                msg: ~"EOF while parsing object"}));
        assert_eq!(from_str("{\"a\""),
            Err(Error {
                line: 1u,
                col: 5u,
                msg: ~"EOF while parsing object"}));
        assert_eq!(from_str("{\"a\" "),
            Err(Error {
                line: 1u,
                col: 6u,
                msg: ~"EOF while parsing object"}));

        assert_eq!(from_str("{\"a\" 1"),
            Err(Error {line: 1u, col: 6u, msg: ~"expected `:`"}));
        assert_eq!(from_str("{\"a\":"),
            Err(Error {line: 1u, col: 6u, msg: ~"EOF while parsing value"}));
        assert_eq!(from_str("{\"a\":1"),
            Err(Error {
                line: 1u,
                col: 7u,
                msg: ~"EOF while parsing object"}));
        assert_eq!(from_str("{\"a\":1 1"),
            Err(Error {line: 1u, col: 8u, msg: ~"expected `,` or `}`"}));
        assert_eq!(from_str("{\"a\":1,"),
            Err(Error {
                line: 1u,
                col: 8u,
                msg: ~"EOF while parsing object"}));

        assert_eq!(from_str("{}").unwrap(), mk_object([]));
        assert_eq!(from_str("{\"a\": 3}").unwrap(),
                  mk_object([(~"a", Number(3.0))]));

        assert_eq!(from_str(
                      "{ \"a\": null, \"b\" : true }").unwrap(),
                  mk_object([
                      (~"a", Null),
                      (~"b", Boolean(true))]));
        assert_eq!(from_str("\n{ \"a\": null, \"b\" : true }\n").unwrap(),
                  mk_object([
                      (~"a", Null),
                      (~"b", Boolean(true))]));
        assert_eq!(from_str(
                      "{\"a\" : 1.0 ,\"b\": [ true ]}").unwrap(),
                  mk_object([
                      (~"a", Number(1.0)),
                      (~"b", List(~[Boolean(true)]))
                  ]));
        assert_eq!(from_str(
                      ~"{" +
                          "\"a\": 1.0, " +
                          "\"b\": [" +
                              "true," +
                              "\"foo\\nbar\", " +
                              "{ \"c\": {\"d\": null} } " +
                          "]" +
                      "}").unwrap(),
                  mk_object([
                      (~"a", Number(1.0)),
                      (~"b", List(~[
                          Boolean(true),
                          String(~"foo\nbar"),
                          mk_object([
                              (~"c", mk_object([(~"d", Null)]))
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
        let mut decoder = Decoder::new(from_str(s).unwrap());
        let v: Outer = Decodable::decode(&mut decoder);
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
        let mut decoder = Decoder::new(from_str("null").unwrap());
        let value: Option<~str> = Decodable::decode(&mut decoder);
        assert_eq!(value, None);

        let mut decoder = Decoder::new(from_str("\"jodhpurs\"").unwrap());
        let value: Option<~str> = Decodable::decode(&mut decoder);
        assert_eq!(value, Some(~"jodhpurs"));
    }

    #[test]
    fn test_decode_enum() {
        let mut decoder = Decoder::new(from_str("\"Dog\"").unwrap());
        let value: Animal = Decodable::decode(&mut decoder);
        assert_eq!(value, Dog);

        let s = "{\"variant\":\"Frog\",\"fields\":[\"Henry\",349]}";
        let mut decoder = Decoder::new(from_str(s).unwrap());
        let value: Animal = Decodable::decode(&mut decoder);
        assert_eq!(value, Frog(~"Henry", 349));
    }

    #[test]
    fn test_decode_map() {
        let s = ~"{\"a\": \"Dog\", \"b\": {\"variant\":\"Frog\",\"fields\":[\"Henry\", 349]}}";
        let mut decoder = Decoder::new(from_str(s).unwrap());
        let mut map: TreeMap<~str, Animal> = Decodable::decode(&mut decoder);

        assert_eq!(map.pop(&~"a"), Some(Dog));
        assert_eq!(map.pop(&~"b"), Some(Frog(~"Henry", 349)));
    }

    #[test]
    fn test_multiline_errors() {
        assert_eq!(from_str("{\n  \"foo\":\n \"bar\""),
            Err(Error {
                line: 3u,
                col: 8u,
                msg: ~"EOF while parsing object"}));
    }

    #[deriving(Decodable)]
    struct DecodeStruct {
        x: f64,
        y: bool,
        z: ~str,
        w: ~[DecodeStruct]
    }
    #[deriving(Decodable)]
    enum DecodeEnum {
        A(f64),
        B(~str)
    }
    fn check_err<T: Decodable<Decoder>>(to_parse: &'static str, expected_error: &str) {
        use std::task;
        let res = task::try(proc() {
            // either fails in `decode` (which is what we want), or
            // returns Some(error_message)/None if the string was
            // invalid or valid JSON.
            match from_str(to_parse) {
                Err(e) => Some(e.to_str()),
                Ok(json) => {
                    let _: T = Decodable::decode(&mut Decoder::new(json));
                    None
                }
            }
        });
        match res {
            Ok(Some(parse_error)) => fail!("`{}` is not valid json: {}",
                                           to_parse, parse_error),
            Ok(None) => fail!("`{}` parsed & decoded ok, expecting error `{}`",
                              to_parse, expected_error),
            Err(e) => {
                let err = e.as_ref::<~str>().unwrap();
                assert!(err.contains(expected_error),
                        "`{}` errored incorrectly, found `{}` expecting `{}`",
                        to_parse, *err, expected_error);
            }
        }
    }
    #[test]
    fn test_decode_errors_struct() {
        check_err::<DecodeStruct>("[]", "object but found list");
        check_err::<DecodeStruct>("{\"x\": true, \"y\": true, \"z\": \"\", \"w\": []}",
                                  "number but found boolean");
        check_err::<DecodeStruct>("{\"x\": 1, \"y\": [], \"z\": \"\", \"w\": []}",
                                  "boolean but found list");
        check_err::<DecodeStruct>("{\"x\": 1, \"y\": true, \"z\": {}, \"w\": []}",
                                  "string but found object");
        check_err::<DecodeStruct>("{\"x\": 1, \"y\": true, \"z\": \"\", \"w\": null}",
                                  "list but found null");
        check_err::<DecodeStruct>("{\"x\": 1, \"y\": true, \"z\": \"\"}",
                                  "'w' field in object");
    }
    #[test]
    fn test_decode_errors_enum() {
        check_err::<DecodeEnum>("{}",
                                "'variant' field in object");
        check_err::<DecodeEnum>("{\"variant\": 1}",
                                "string but found number");
        check_err::<DecodeEnum>("{\"variant\": \"A\"}",
                                "'fields' field in object");
        check_err::<DecodeEnum>("{\"variant\": \"A\", \"fields\": null}",
                                "list but found null");
        check_err::<DecodeEnum>("{\"variant\": \"C\", \"fields\": []}",
                                "unknown variant name");
    }
}
