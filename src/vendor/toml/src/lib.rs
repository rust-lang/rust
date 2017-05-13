//! A TOML-parsing library
//!
//! This library is an implementation in Rust of a parser for TOML configuration
//! files [1]. It is focused around high quality errors including specific spans
//! and detailed error messages when things go wrong.
//!
//! This implementation currently passes the language agnostic [test suite][2].
//!
//! # Example
//!
//! ```
//! let toml = r#"
//!     [test]
//!     foo = "bar"
//! "#;
//!
//! let value = toml::Parser::new(toml).parse().unwrap();
//! println!("{:?}", value);
//! ```
//!
//! # Conversions
//!
//! This library also supports using the standard `Encodable` and `Decodable`
//! traits with TOML values. This library provides the following conversion
//! capabilities:
//!
//! * `String` => `toml::Value` - via `Parser`
//! * `toml::Value` => `String` - via `Display`
//! * `toml::Value` => rust object - via `Decoder`
//! * rust object => `toml::Value` - via `Encoder`
//!
//! Convenience functions for performing multiple conversions at a time are also
//! provided.
//!
//! [1]: https://github.com/mojombo/toml
//! [2]: https://github.com/BurntSushi/toml-test

#![doc(html_root_url = "http://alexcrichton.com/toml-rs")]
#![deny(missing_docs)]
#![cfg_attr(test, deny(warnings))]

#[cfg(feature = "rustc-serialize")] extern crate rustc_serialize;
#[cfg(feature = "serde")] extern crate serde;

use std::collections::BTreeMap;
use std::str::FromStr;

pub use parser::{Parser, ParserError};

#[cfg(any(feature = "rustc-serialize", feature = "serde"))]
pub use self::encoder::{Encoder, Error, encode, encode_str};
#[cfg(any(feature = "rustc-serialize", feature = "serde"))]
pub use self::decoder::{Decoder, DecodeError, DecodeErrorKind, decode, decode_str};

mod parser;
mod display;
#[cfg(any(feature = "rustc-serialize", feature = "serde"))]
mod encoder;
#[cfg(any(feature = "rustc-serialize", feature = "serde"))]
mod decoder;

/// Representation of a TOML value.
#[derive(PartialEq, Clone, Debug)]
#[allow(missing_docs)]
pub enum Value {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Datetime(String),
    Array(Array),
    Table(Table),
}

/// Type representing a TOML array, payload of the Value::Array variant
pub type Array = Vec<Value>;

/// Type representing a TOML table, payload of the Value::Table variant
pub type Table = BTreeMap<String, Value>;

impl Value {
    /// Tests whether this and another value have the same type.
    pub fn same_type(&self, other: &Value) -> bool {
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

    /// Returns a human-readable representation of the type of this value.
    pub fn type_str(&self) -> &'static str {
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

    /// Extracts the string of this value if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match *self { Value::String(ref s) => Some(&**s), _ => None }
    }

    /// Extracts the integer value if it is an integer.
    pub fn as_integer(&self) -> Option<i64> {
        match *self { Value::Integer(i) => Some(i), _ => None }
    }

    /// Extracts the float value if it is a float.
    pub fn as_float(&self) -> Option<f64> {
        match *self { Value::Float(f) => Some(f), _ => None }
    }

    /// Extracts the boolean value if it is a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match *self { Value::Boolean(b) => Some(b), _ => None }
    }

    /// Extracts the datetime value if it is a datetime.
    ///
    /// Note that a parsed TOML value will only contain ISO 8601 dates. An
    /// example date is:
    ///
    /// ```notrust
    /// 1979-05-27T07:32:00Z
    /// ```
    pub fn as_datetime(&self) -> Option<&str> {
        match *self { Value::Datetime(ref s) => Some(&**s), _ => None }
    }

    /// Extracts the array value if it is an array.
    pub fn as_slice(&self) -> Option<&[Value]> {
        match *self { Value::Array(ref s) => Some(&**s), _ => None }
    }

    /// Extracts the table value if it is a table.
    pub fn as_table(&self) -> Option<&Table> {
        match *self { Value::Table(ref s) => Some(s), _ => None }
    }

    /// Lookups for value at specified path.
    ///
    /// Uses '.' as a path separator.
    ///
    /// Note: arrays have zero-based indexes.
    ///
    /// Note: empty path returns self.
    ///
    /// ```
    /// # #![allow(unstable)]
    /// let toml = r#"
    ///      [test]
    ///      foo = "bar"
    ///
    ///      [[values]]
    ///      foo = "baz"
    ///
    ///      [[values]]
    ///      foo = "qux"
    /// "#;
    /// let value: toml::Value = toml.parse().unwrap();
    ///
    /// let foo = value.lookup("test.foo").unwrap();
    /// assert_eq!(foo.as_str().unwrap(), "bar");
    ///
    /// let foo = value.lookup("values.1.foo").unwrap();
    /// assert_eq!(foo.as_str().unwrap(), "qux");
    ///
    /// let no_bar = value.lookup("test.bar");
    /// assert_eq!(no_bar.is_none(), true);
    /// ```
    pub fn lookup<'a>(&'a self, path: &'a str) -> Option<&'a Value> {
        let ref path = match Parser::new(path).lookup() {
            Some(path) => path,
            None => return None,
        };
        let mut cur_value = self;
        if path.len() == 0 {
            return Some(cur_value)
        }

        for key in path {
            match *cur_value {
                Value::Table(ref hm) => {
                    match hm.get(key) {
                        Some(v) => cur_value = v,
                        None => return None
                    }
                },
                Value::Array(ref v) => {
                    match key.parse::<usize>().ok() {
                        Some(idx) if idx < v.len() => cur_value = &v[idx],
                        _ => return None
                    }
                },
                _ => return None
            }
        };

        Some(cur_value)

    }
    /// Lookups for mutable value at specified path.
    ///
    /// Uses '.' as a path separator.
    ///
    /// Note: arrays have zero-based indexes.
    ///
    /// Note: empty path returns self.
    ///
    /// ```
    /// # #![allow(unstable)]
    /// let toml = r#"
    ///      [test]
    ///      foo = "bar"
    ///
    ///      [[values]]
    ///      foo = "baz"
    ///
    ///      [[values]]
    ///      foo = "qux"
    /// "#;
    /// let mut value: toml::Value = toml.parse().unwrap();
    /// {
    ///    let string = value.lookup_mut("test.foo").unwrap();
    ///    assert_eq!(string, &mut toml::Value::String(String::from("bar")));
    ///    *string = toml::Value::String(String::from("foo"));
    /// }
    /// let result = value.lookup_mut("test.foo").unwrap();
    /// assert_eq!(result.as_str().unwrap(), "foo");
    /// ```
    pub fn lookup_mut(&mut self, path: &str) -> Option<&mut Value> {
       let ref path = match Parser::new(path).lookup() {
            Some(path) => path,
            None => return None,
        };

        let mut cur = self;
        if path.len() == 0 {
            return Some(cur)
        }

        for key in path {
            let tmp = cur;
            match *tmp {
                Value::Table(ref mut hm) => {
                    match hm.get_mut(key) {
                        Some(v) => cur = v,
                        None => return None
                    }
                }
                Value::Array(ref mut v) => {
                    match key.parse::<usize>().ok() {
                        Some(idx) if idx < v.len() => cur = &mut v[idx],
                        _ => return None
                    }
                }
                _ => return None
           }
        }
        Some(cur)
    }
}

impl FromStr for Value {
    type Err = Vec<ParserError>;
    fn from_str(s: &str) -> Result<Value, Vec<ParserError>> {
        let mut p = Parser::new(s);
        match p.parse().map(Value::Table) {
            Some(n) => Ok(n),
            None => Err(p.errors),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Value;

    #[test]
    fn lookup_mut_change() {
        let toml = r#"
              [test]
              foo = "bar"

              [[values]]
              foo = "baz"

              [[values]]
              foo = "qux"
        "#;

        let mut value: Value = toml.parse().unwrap();
        {
          let foo = value.lookup_mut("values.0.foo").unwrap();
          *foo = Value::String(String::from("bar"));
        }
        let foo = value.lookup("values.0.foo").unwrap();
        assert_eq!(foo.as_str().unwrap(), "bar");
    }

    #[test]
    fn lookup_mut_valid() {
        let toml = r#"
              [test]
              foo = "bar"

              [[values]]
              foo = "baz"

              [[values]]
              foo = "qux"
        "#;

        let mut value: Value = toml.parse().unwrap();

        {
            let test_foo = value.lookup_mut("test.foo").unwrap();
            assert_eq!(test_foo.as_str().unwrap(), "bar");
        }

        {
            let foo1 = value.lookup_mut("values.1.foo").unwrap();
            assert_eq!(foo1.as_str().unwrap(), "qux");
        }

        assert!(value.lookup_mut("test.bar").is_none());
        assert!(value.lookup_mut("test.foo.bar").is_none());
    }

    #[test]
    fn lookup_mut_invalid_index() {
        let toml = r#"
            [[values]]
            foo = "baz"
        "#;

        let mut value: Value = toml.parse().unwrap();

        {
            let foo = value.lookup_mut("test.foo");
            assert!(foo.is_none());
        }

        {
            let foo = value.lookup_mut("values.100.foo");
            assert!(foo.is_none());
        }

        {
            let foo = value.lookup_mut("values.str.foo");
            assert!(foo.is_none());
        }
    }

    #[test]
    fn lookup_mut_self() {
        let mut value: Value = r#"foo = "bar""#.parse().unwrap();

        {
            let foo = value.lookup_mut("foo").unwrap();
            assert_eq!(foo.as_str().unwrap(), "bar");
        }

        let foo = value.lookup_mut("").unwrap();
        assert!(foo.as_table().is_some());

        let baz = foo.lookup_mut("foo").unwrap();
        assert_eq!(baz.as_str().unwrap(), "bar");
    }

    #[test]
    fn lookup_valid() {
        let toml = r#"
              [test]
              foo = "bar"

              [[values]]
              foo = "baz"

              [[values]]
              foo = "qux"
        "#;

        let value: Value = toml.parse().unwrap();

        let test_foo = value.lookup("test.foo").unwrap();
        assert_eq!(test_foo.as_str().unwrap(), "bar");

        let foo1 = value.lookup("values.1.foo").unwrap();
        assert_eq!(foo1.as_str().unwrap(), "qux");

        assert!(value.lookup("test.bar").is_none());
        assert!(value.lookup("test.foo.bar").is_none());
    }

    #[test]
    fn lookup_invalid_index() {
        let toml = r#"
            [[values]]
            foo = "baz"
        "#;

        let value: Value = toml.parse().unwrap();

        let foo = value.lookup("test.foo");
        assert!(foo.is_none());

        let foo = value.lookup("values.100.foo");
        assert!(foo.is_none());

        let foo = value.lookup("values.str.foo");
        assert!(foo.is_none());
    }

    #[test]
    fn lookup_self() {
        let value: Value = r#"foo = "bar""#.parse().unwrap();

        let foo = value.lookup("foo").unwrap();
        assert_eq!(foo.as_str().unwrap(), "bar");

        let foo = value.lookup("").unwrap();
        assert!(foo.as_table().is_some());

        let baz = foo.lookup("foo").unwrap();
        assert_eq!(baz.as_str().unwrap(), "bar");
    }

    #[test]
    fn lookup_advanced() {
        let value: Value = "[table]\n\"value\" = 0".parse().unwrap();
        let looked = value.lookup("table.\"value\"").unwrap();
        assert_eq!(*looked, Value::Integer(0));
    }

    #[test]
    fn lookup_advanced_table() {
        let value: Value = "[table.\"name.other\"]\nvalue = \"my value\"".parse().unwrap();
        let looked = value.lookup(r#"table."name.other".value"#).unwrap();
        assert_eq!(*looked, Value::String(String::from("my value")));
    }

    #[test]
    fn lookup_mut_advanced() {
        let mut value: Value = "[table]\n\"value\" = [0, 1, 2]".parse().unwrap();
        let looked = value.lookup_mut("table.\"value\".1").unwrap();
        assert_eq!(*looked, Value::Integer(1));
    }

    #[test]
    fn single_dot() {
        let value: Value = "[table]\n\"value\" = [0, 1, 2]".parse().unwrap();
        assert_eq!(None, value.lookup("."));
    }

    #[test]
    fn array_dot() {
        let value: Value = "[table]\n\"value\" = [0, 1, 2]".parse().unwrap();
        assert_eq!(None, value.lookup("0."));
    }

    #[test]
    fn dot_inside() {
        let value: Value = "[table]\n\"value\" = [0, 1, 2]".parse().unwrap();
        assert_eq!(None, value.lookup("table.\"value.0\""));
    }

    #[test]
    fn table_with_quotes() {
        let value: Value = "[table.\"element\"]\n\"value\" = [0, 1, 2]".parse().unwrap();
        assert_eq!(None, value.lookup("\"table.element\".\"value\".0"));
    }

    #[test]
    fn table_with_quotes_2() {
        let value: Value = "[table.\"element\"]\n\"value\" = [0, 1, 2]".parse().unwrap();
        assert_eq!(Value::Integer(0), *value.lookup("table.\"element\".\"value\".0").unwrap());
    }

}
