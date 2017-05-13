use std::collections::BTreeMap;
use std::error;
use std::fmt;
use std::mem;

use {Value, Table};

#[cfg(feature = "rustc-serialize")] mod rustc_serialize;
#[cfg(feature = "serde")] mod serde;

/// A structure to transform Rust values into TOML values.
///
/// This encoder implements the serialization `Encoder` interface, allowing
/// `Encodable` rust types to be fed into the encoder. The output of this
/// encoder is a TOML `Table` structure. The resulting TOML can be stringified
/// if necessary.
///
/// # Example
///
/// ```
/// extern crate rustc_serialize;
/// extern crate toml;
///
/// # fn main() {
/// use toml::{Encoder, Value};
/// use rustc_serialize::Encodable;
///
/// #[derive(RustcEncodable)]
/// struct MyStruct { foo: isize, bar: String }
/// let my_struct = MyStruct { foo: 4, bar: "hello!".to_string() };
///
/// let mut e = Encoder::new();
/// my_struct.encode(&mut e).unwrap();
///
/// assert_eq!(e.toml.get(&"foo".to_string()), Some(&Value::Integer(4)))
/// # }
/// ```
pub struct Encoder {
    /// Output TOML that is emitted. The current version of this encoder forces
    /// the top-level representation of a structure to be a table.
    ///
    /// This field can be used to extract the return value after feeding a value
    /// into this `Encoder`.
    pub toml: Table,
    state: State,
}

/// Enumeration of errors which can occur while encoding a rust value into a
/// TOML value.
#[allow(missing_copy_implementations)]
#[derive(Debug)]
pub enum Error {
    /// Indication that a key was needed when a value was emitted, but no key
    /// was previously emitted.
    NeedsKey,
    /// Indication that a key was emitted, but not value was emitted.
    NoValue,
    /// Indicates that a map key was attempted to be emitted at an invalid
    /// location.
    InvalidMapKeyLocation,
    /// Indicates that a type other than a string was attempted to be used as a
    /// map key type.
    InvalidMapKeyType,
    /// A custom error type was generated
    Custom(String),
}

#[derive(PartialEq)]
enum State {
    Start,
    NextKey(String),
    NextArray(Vec<Value>),
    NextMapKey,
}

impl Encoder {
    /// Constructs a new encoder which will emit to the given output stream.
    pub fn new() -> Encoder {
        Encoder { state: State::Start, toml: BTreeMap::new() }
    }

    fn emit_value(&mut self, v: Value) -> Result<(), Error> {
        match mem::replace(&mut self.state, State::Start) {
            State::NextKey(key) => { self.toml.insert(key, v); Ok(()) }
            State::NextArray(mut vec) => {
                // TODO: validate types
                vec.push(v);
                self.state = State::NextArray(vec);
                Ok(())
            }
            State::NextMapKey => {
                match v {
                    Value::String(s) => { self.state = State::NextKey(s); Ok(()) }
                    _ => Err(Error::InvalidMapKeyType)
                }
            }
            _ => Err(Error::NeedsKey)
        }
    }

    fn emit_none(&mut self) -> Result<(), Error> {
        match mem::replace(&mut self.state, State::Start) {
            State::Start => unreachable!(),
            State::NextKey(_) => Ok(()),
            State::NextArray(..) => panic!("how to encode None in an array?"),
            State::NextMapKey => Err(Error::InvalidMapKeyLocation),
        }
    }

    fn seq<F>(&mut self, f: F) -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        let old = mem::replace(&mut self.state, State::NextArray(Vec::new()));
        try!(f(self));
        match mem::replace(&mut self.state, old) {
            State::NextArray(v) => self.emit_value(Value::Array(v)),
            _ => unreachable!(),
        }
    }

    fn table<F>(&mut self, f: F) -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        match mem::replace(&mut self.state, State::Start) {
            State::NextKey(key) => {
                let mut nested = Encoder::new();
                try!(f(&mut nested));
                self.toml.insert(key, Value::Table(nested.toml));
                Ok(())
            }
            State::NextArray(mut arr) => {
                let mut nested = Encoder::new();
                try!(f(&mut nested));
                arr.push(Value::Table(nested.toml));
                self.state = State::NextArray(arr);
                Ok(())
            }
            State::Start => f(self),
            State::NextMapKey => Err(Error::InvalidMapKeyLocation),
        }
    }

    fn table_key<F>(&mut self, f: F) -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        match mem::replace(&mut self.state, State::NextMapKey) {
            State::Start => {}
            _ => return Err(Error::InvalidMapKeyLocation),
        }
        try!(f(self));
        match self.state {
            State::NextKey(_) => Ok(()),
            _ => Err(Error::InvalidMapKeyLocation),
        }
    }
}

/// Encodes an encodable value into a TOML value.
///
/// This function expects the type given to represent a TOML table in some form.
/// If encoding encounters an error, then this function will fail the task.
#[cfg(feature = "rustc-serialize")]
pub fn encode<T: ::rustc_serialize::Encodable>(t: &T) -> Value {
    let mut e = Encoder::new();
    t.encode(&mut e).unwrap();
    Value::Table(e.toml)
}

/// Encodes an encodable value into a TOML value.
///
/// This function expects the type given to represent a TOML table in some form.
/// If encoding encounters an error, then this function will fail the task.
#[cfg(all(not(feature = "rustc-serialize"), feature = "serde"))]
pub fn encode<T: ::serde::Serialize>(t: &T) -> Value {
    let mut e = Encoder::new();
    t.serialize(&mut e).unwrap();
    Value::Table(e.toml)
}

/// Encodes an encodable value into a TOML string.
///
/// This function expects the type given to represent a TOML table in some form.
/// If encoding encounters an error, then this function will fail the task.
#[cfg(feature = "rustc-serialize")]
pub fn encode_str<T: ::rustc_serialize::Encodable>(t: &T) -> String {
    encode(t).to_string()
}

/// Encodes an encodable value into a TOML string.
///
/// This function expects the type given to represent a TOML table in some form.
/// If encoding encounters an error, then this function will fail the task.
#[cfg(all(not(feature = "rustc-serialize"), feature = "serde"))]
pub fn encode_str<T: ::serde::Serialize>(t: &T) -> String {
    encode(t).to_string()
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::NeedsKey => write!(f, "need a key to encode"),
            Error::NoValue => write!(f, "no value to emit for a previous key"),
            Error::InvalidMapKeyLocation => write!(f, "a map cannot be emitted \
                                                       at this location"),
            Error::InvalidMapKeyType => write!(f, "only strings can be used as \
                                                   key types"),
            Error::Custom(ref s) => write!(f, "custom error: {}", s),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str { "TOML encoding error" }
}
