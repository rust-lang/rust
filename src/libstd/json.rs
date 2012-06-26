// Rust JSON serialization library
// Copyright (c) 2011 Google Inc.

#[doc = "json serialization"];

import result::{result, ok, err};
import io;
import io::{reader_util, writer_util};
import map;
import map::hashmap;

export json;
export error;
export to_writer;
export to_str;
export from_reader;
export from_str;
export eq;
export to_json;

export num;
export string;
export boolean;
export list;
export dict;
export null;

#[doc = "Represents a json value"]
enum json {
    num(float),
    string(@str),
    boolean(bool),
    list(@~[json]),
    dict(map::hashmap<str, json>),
    null,
}

type error = {
    line: uint,
    col: uint,
    msg: @str,
};

#[doc = "Serializes a json value into a io::writer"]
fn to_writer(wr: io::writer, j: json) {
    alt j {
      num(n) { wr.write_str(float::to_str(n, 6u)); }
      string(s) {
        wr.write_str(escape_str(*s));
      }
      boolean(b) {
        wr.write_str(if b { "true" } else { "false" });
      }
      list(v) {
        wr.write_char('[');
        let mut first = true;
        for (*v).each { |item|
            if !first {
                wr.write_str(", ");
            }
            first = false;
            to_writer(wr, item);
        };
        wr.write_char(']');
      }
      dict(d) {
        if d.size() == 0u {
            wr.write_str("{}");
            ret;
        }

        wr.write_str("{ ");
        let mut first = true;
        for d.each { |key, value|
            if !first {
                wr.write_str(", ");
            }
            first = false;
            wr.write_str(escape_str(key));
            wr.write_str(": ");
            to_writer(wr, value);
        };
        wr.write_str(" }");
      }
      null {
        wr.write_str("null");
      }
    }
}

fn escape_str(s: str) -> str {
    let mut escaped = "\"";
    do str::chars_iter(s) { |c|
        alt c {
          '"' { escaped += "\\\""; }
          '\\' { escaped += "\\\\"; }
          '\x08' { escaped += "\\b"; }
          '\x0c' { escaped += "\\f"; }
          '\n' { escaped += "\\n"; }
          '\r' { escaped += "\\r"; }
          '\t' { escaped += "\\t"; }
          _ { escaped += str::from_char(c); }
        }
    };

    escaped += "\"";

    escaped
}

#[doc = "Serializes a json value into a string"]
fn to_str(j: json) -> str {
    io::with_str_writer({ |wr| to_writer(wr, j) })
}

type parser = {
    rdr: io::reader,
    mut ch: char,
    mut line: uint,
    mut col: uint,
};

impl parser for parser {
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

    fn error<T>(+msg: str) -> result<T, error> {
        err({ line: self.line, col: self.col, msg: @msg })
    }

    fn parse() -> result<json, error> {
        alt self.parse_value() {
          ok(value) {
            // Skip trailing whitespaces.
            self.parse_whitespace();
            // Make sure there is no trailing characters.
            if self.eof() {
                ok(value)
            } else {
                self.error("trailing characters")
            }
          }
          e { e }
        }
    }

    fn parse_value() -> result<json, error> {
        self.parse_whitespace();

        if self.eof() { ret self.error("EOF while parsing value"); }

        alt self.ch {
          'n' { self.parse_ident("ull", null) }
          't' { self.parse_ident("rue", boolean(true)) }
          'f' { self.parse_ident("alse", boolean(false)) }
          '0' to '9' | '-' { self.parse_number() }
          '"' {
              alt self.parse_str() {
                ok(s) { ok(string(s)) }
                err(e) { err(e) }
              }
          }
          '[' { self.parse_list() }
          '{' { self.parse_object() }
          _ { self.error("invalid syntax") }
        }
    }

    fn parse_whitespace() {
        while char::is_whitespace(self.ch) { self.bump(); }
    }

    fn parse_ident(ident: str, value: json) -> result<json, error> {
        if str::all(ident, { |c| c == self.next_char() }) {
            self.bump();
            ok(value)
        } else {
            self.error("invalid syntax")
        }
    }

    fn parse_number() -> result<json, error> {
        let mut neg = 1f;

        if self.ch == '-' {
            self.bump();
            neg = -1f;
        }

        let mut res = alt self.parse_integer() {
          ok(res) { res }
          err(e) { ret err(e); }
        };

        if self.ch == '.' {
            alt self.parse_decimal(res) {
              ok(r) { res = r; }
              err(e) { ret err(e); }
            }
        }

        if self.ch == 'e' || self.ch == 'E' {
            alt self.parse_exponent(res) {
              ok(r) { res = r; }
              err(e) { ret err(e); }
            }
        }

        ok(num(neg * res))
    }

    fn parse_integer() -> result<float, error> {
        let mut res = 0f;

        alt self.ch {
          '0' {
            self.bump();

            // There can be only one leading '0'.
            alt self.ch {
              '0' to '9' { ret self.error("invalid number"); }
              _ {}
            }
          }
          '1' to '9' {
            while !self.eof() {
                alt self.ch {
                  '0' to '9' {
                    res *= 10f;
                    res += ((self.ch as int) - ('0' as int)) as float;

                    self.bump();
                  }
                  _ { break; }
                }
            }
          }
          _ { ret self.error("invalid number"); }
        }

        ok(res)
    }

    fn parse_decimal(res: float) -> result<float, error> {
        self.bump();

        // Make sure a digit follows the decimal place.
        alt self.ch {
          '0' to '9' {}
          _ { ret self.error("invalid number"); }
        }

        let mut res = res;
        let mut dec = 1f;
        while !self.eof() {
            alt self.ch {
              '0' to '9' {
                dec /= 10f;
                res += (((self.ch as int) - ('0' as int)) as float) * dec;

                self.bump();
              }
              _ { break; }
            }
        }

        ok(res)
    }

    fn parse_exponent(res: float) -> result<float, error> {
        self.bump();

        let mut res = res;
        let mut exp = 0u;
        let mut neg_exp = false;

        alt self.ch {
          '+' { self.bump(); }
          '-' { self.bump(); neg_exp = true; }
          _ {}
        }

        // Make sure a digit follows the exponent place.
        alt self.ch {
          '0' to '9' {}
          _ { ret self.error("invalid number"); }
        }

        while !self.eof() {
            alt self.ch {
              '0' to '9' {
                exp *= 10u;
                exp += (self.ch as uint) - ('0' as uint);

                self.bump();
              }
              _ { break; }
            }
        }

        let exp = float::pow_with_uint(10u, exp);
        if neg_exp {
            res /= exp;
        } else {
            res *= exp;
        }

        ok(res)
    }

    fn parse_str() -> result<@str, error> {
        let mut escape = false;
        let mut res = "";

        while !self.eof() {
            self.bump();

            if (escape) {
                alt self.ch {
                  '"' { str::push_char(res, '"'); }
                  '\\' { str::push_char(res, '\\'); }
                  '/' { str::push_char(res, '/'); }
                  'b' { str::push_char(res, '\x08'); }
                  'f' { str::push_char(res, '\x0c'); }
                  'n' { str::push_char(res, '\n'); }
                  'r' { str::push_char(res, '\r'); }
                  't' { str::push_char(res, '\t'); }
                  'u' {
                      // Parse \u1234.
                      let mut i = 0u;
                      let mut n = 0u;
                      while i < 4u {
                          alt self.next_char() {
                            '0' to '9' {
                              n = n * 10u +
                                  (self.ch as uint) - ('0' as uint);
                            }
                            _ { ret self.error("invalid \\u escape"); }
                          }
                          i += 1u;
                      }

                      // Error out if we didn't parse 4 digits.
                      if i != 4u {
                          ret self.error("invalid \\u escape");
                      }

                      str::push_char(res, n as char);
                  }
                  _ { ret self.error("invalid escape"); }
                }
                escape = false;
            } else if self.ch == '\\' {
                escape = true;
            } else {
                if self.ch == '"' {
                    self.bump();
                    ret ok(@res);
                }
                str::push_char(res, self.ch);
            }
        }

        self.error("EOF while parsing string")
    }

    fn parse_list() -> result<json, error> {
        self.bump();
        self.parse_whitespace();

        let mut values = ~[];

        if self.ch == ']' {
            self.bump();
            ret ok(list(@values));
        }

        loop {
            alt self.parse_value() {
              ok(v) { vec::push(values, v); }
              e { ret e; }
            }

            self.parse_whitespace();
            if self.eof() {
                ret self.error("EOF while parsing list");
            }

            alt self.ch {
              ',' { self.bump(); }
              ']' { self.bump(); ret ok(list(@values)); }
              _ { ret self.error("expecting ',' or ']'"); }
            }
        };
    }

    fn parse_object() -> result<json, error> {
        self.bump();
        self.parse_whitespace();

        let values = map::str_hash();

        if self.ch == '}' {
          self.bump();
          ret ok(dict(values));
        }

        while !self.eof() {
            self.parse_whitespace();

            if self.ch != '"' {
                ret self.error("key must be a string");
            }

            let key = alt self.parse_str() {
              ok(key) { key }
              err(e) { ret err(e); }
            };

            self.parse_whitespace();

            if self.ch != ':' {
                if self.eof() { break; }
                ret self.error("expecting ':'");
            }
            self.bump();

            alt self.parse_value() {
              ok(value) { values.insert(copy *key, value); }
              e { ret e; }
            }
            self.parse_whitespace();

            alt self.ch {
              ',' { self.bump(); }
              '}' { self.bump(); ret ok(dict(values)); }
              _ {
                  if self.eof() { break; }
                  ret self.error("expecting ',' or '}'");
              }
            }
        }

        ret self.error("EOF while parsing object");
    }
}

#[doc = "Deserializes a json value from an io::reader"]
fn from_reader(rdr: io::reader) -> result<json, error> {
    let parser = {
        rdr: rdr,
        mut ch: rdr.read_char(),
        mut line: 1u,
        mut col: 1u,
    };

    parser.parse()
}

#[doc = "Deserializes a json value from a string"]
fn from_str(s: str) -> result<json, error> {
    io::with_str_reader(s, from_reader)
}

#[doc = "Test if two json values are equal"]
fn eq(value0: json, value1: json) -> bool {
    alt (value0, value1) {
      (num(f0), num(f1)) { f0 == f1 }
      (string(s0), string(s1)) { s0 == s1 }
      (boolean(b0), boolean(b1)) { b0 == b1 }
      (list(l0), list(l1)) { vec::all2(*l0, *l1, eq) }
      (dict(d0), dict(d1)) {
          if d0.size() == d1.size() {
              let mut equal = true;
              for d0.each { |k, v0|
                  alt d1.find(k) {
                    some(v1) {
                        if !eq(v0, v1) { equal = false; } }
                    none { equal = false; }
                  }
              };
              equal
          } else {
              false
          }
      }
      (null, null) { true }
      _ { false }
    }
}

iface to_json { fn to_json() -> json; }

impl of to_json for json {
    fn to_json() -> json { self }
}

impl of to_json for @json {
    fn to_json() -> json { *self }
}

impl of to_json for int {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for i8 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for i16 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for i32 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for i64 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for uint {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for u8 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for u16 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for u32 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for u64 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for float {
    fn to_json() -> json { num(self) }
}

impl of to_json for f32 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for f64 {
    fn to_json() -> json { num(self as float) }
}

impl of to_json for () {
    fn to_json() -> json { null }
}

impl of to_json for bool {
    fn to_json() -> json { boolean(self) }
}

impl of to_json for str {
    fn to_json() -> json { string(@copy self) }
}

impl of to_json for @str {
    fn to_json() -> json { string(self) }
}

impl <A: to_json copy, B: to_json copy> of to_json for (A, B) {
    fn to_json() -> json {
        let (a, b) = self;
        list(@~[a.to_json(), b.to_json()])
    }
}

impl <A: to_json copy, B: to_json copy, C: to_json copy>
  of to_json for (A, B, C) {
    fn to_json() -> json {
        let (a, b, c) = self;
        list(@~[a.to_json(), b.to_json(), c.to_json()])
    }
}

impl <A: to_json> of to_json for ~[A] {
    fn to_json() -> json { list(@self.map({ |elt| elt.to_json() })) }
}

impl <A: to_json copy> of to_json for hashmap<str, A> {
    fn to_json() -> json {
        let d = map::str_hash();
        for self.each() { |key, value|
            d.insert(copy key, value.to_json());
        }
        dict(d)
    }
}

impl <A: to_json> of to_json for option<A> {
    fn to_json() -> json {
        alt self {
          none { null }
          some(value) { value.to_json() }
        }
    }
}

impl of to_str::to_str for json {
    fn to_str() -> str { to_str(self) }
}

impl of to_str::to_str for error {
    fn to_str() -> str {
        #fmt("%u:%u: %s", self.line, self.col, *self.msg)
    }
}

#[cfg(test)]
mod tests {
    fn mk_dict(items: ~[(str, json)]) -> json {
        let d = map::str_hash();

        do vec::iter(items) { |item|
            let (key, value) = copy item;
            d.insert(key, value);
        };

        dict(d)
    }

    #[test]
    fn test_write_null() {
        assert to_str(null) == "null";
    }

    #[test]
    fn test_write_num() {
        assert to_str(num(3f)) == "3";
        assert to_str(num(3.1f)) == "3.1";
        assert to_str(num(-1.5f)) == "-1.5";
        assert to_str(num(0.5f)) == "0.5";
    }

    #[test]
    fn test_write_str() {
        assert to_str(string(@"")) == "\"\"";
        assert to_str(string(@"foo")) == "\"foo\"";
    }

    #[test]
    fn test_write_bool() {
        assert to_str(boolean(true)) == "true";
        assert to_str(boolean(false)) == "false";
    }

    #[test]
    fn test_write_list() {
        assert to_str(list(@~[])) == "[]";
        assert to_str(list(@~[boolean(true)])) == "[true]";
        assert to_str(list(@~[
            boolean(false),
            null,
            list(@~[string(@"foo\nbar"), num(3.5f)])
        ])) == "[false, null, [\"foo\\nbar\", 3.5]]";
    }

    #[test]
    fn test_write_dict() {
        assert to_str(mk_dict(~[])) == "{}";
        assert to_str(mk_dict(~[("a", boolean(true))])) == "{ \"a\": true }";
        assert to_str(mk_dict(~[
            ("a", boolean(true)),
            ("b", list(@~[
                mk_dict(~[("c", string(@"\x0c\r"))]),
                mk_dict(~[("d", string(@""))])
            ]))
        ])) ==
            "{ " +
                "\"a\": true, " +
                "\"b\": [" +
                    "{ \"c\": \"\\f\\r\" }, " +
                    "{ \"d\": \"\" }" +
                "]" +
            " }";
    }

    #[test]
    fn test_trailing_characters() {
        assert from_str("nulla") ==
            err({line: 1u, col: 5u, msg: @"trailing characters"});
        assert from_str("truea") ==
            err({line: 1u, col: 5u, msg: @"trailing characters"});
        assert from_str("falsea") ==
            err({line: 1u, col: 6u, msg: @"trailing characters"});
        assert from_str("1a") ==
            err({line: 1u, col: 2u, msg: @"trailing characters"});
        assert from_str("[]a") ==
            err({line: 1u, col: 3u, msg: @"trailing characters"});
        assert from_str("{}a") ==
            err({line: 1u, col: 3u, msg: @"trailing characters"});
    }

    #[test]
    fn test_read_identifiers() {
        assert from_str("n") ==
            err({line: 1u, col: 2u, msg: @"invalid syntax"});
        assert from_str("nul") ==
            err({line: 1u, col: 4u, msg: @"invalid syntax"});

        assert from_str("t") ==
            err({line: 1u, col: 2u, msg: @"invalid syntax"});
        assert from_str("truz") ==
            err({line: 1u, col: 4u, msg: @"invalid syntax"});

        assert from_str("f") ==
            err({line: 1u, col: 2u, msg: @"invalid syntax"});
        assert from_str("faz") ==
            err({line: 1u, col: 3u, msg: @"invalid syntax"});

        assert from_str("null") == ok(null);
        assert from_str("true") == ok(boolean(true));
        assert from_str("false") == ok(boolean(false));
        assert from_str(" null ") == ok(null);
        assert from_str(" true ") == ok(boolean(true));
        assert from_str(" false ") == ok(boolean(false));
    }

    #[test]
    fn test_read_num() {
        assert from_str("+") ==
            err({line: 1u, col: 1u, msg: @"invalid syntax"});
        assert from_str(".") ==
            err({line: 1u, col: 1u, msg: @"invalid syntax"});

        assert from_str("-") ==
            err({line: 1u, col: 2u, msg: @"invalid number"});
        assert from_str("00") ==
            err({line: 1u, col: 2u, msg: @"invalid number"});
        assert from_str("1.") ==
            err({line: 1u, col: 3u, msg: @"invalid number"});
        assert from_str("1e") ==
            err({line: 1u, col: 3u, msg: @"invalid number"});
        assert from_str("1e+") ==
            err({line: 1u, col: 4u, msg: @"invalid number"});

        assert from_str("3") == ok(num(3f));
        assert from_str("3.1") == ok(num(3.1f));
        assert from_str("-1.2") == ok(num(-1.2f));
        assert from_str("0.4") == ok(num(0.4f));
        assert from_str("0.4e5") == ok(num(0.4e5f));
        assert from_str("0.4e+15") == ok(num(0.4e15f));
        assert from_str("0.4e-01") == ok(num(0.4e-01f));
        assert from_str(" 3 ") == ok(num(3f));
    }

    #[test]
    fn test_read_str() {
        assert from_str("\"") ==
            err({line: 1u, col: 2u, msg: @"EOF while parsing string"});
        assert from_str("\"lol") ==
            err({line: 1u, col: 5u, msg: @"EOF while parsing string"});

        assert from_str("\"\"") == ok(string(@""));
        assert from_str("\"foo\"") == ok(string(@"foo"));
        assert from_str("\"\\\"\"") == ok(string(@"\""));
        assert from_str("\"\\b\"") == ok(string(@"\x08"));
        assert from_str("\"\\n\"") == ok(string(@"\n"));
        assert from_str("\"\\r\"") == ok(string(@"\r"));
        assert from_str("\"\\t\"") == ok(string(@"\t"));
        assert from_str(" \"foo\" ") == ok(string(@"foo"));
    }

    #[test]
    fn test_read_list() {
        assert from_str("[") ==
            err({line: 1u, col: 2u, msg: @"EOF while parsing value"});
        assert from_str("[1") ==
            err({line: 1u, col: 3u, msg: @"EOF while parsing list"});
        assert from_str("[1,") ==
            err({line: 1u, col: 4u, msg: @"EOF while parsing value"});
        assert from_str("[1,]") ==
            err({line: 1u, col: 4u, msg: @"invalid syntax"});
        assert from_str("[6 7]") ==
            err({line: 1u, col: 4u, msg: @"expecting ',' or ']'"});

        assert from_str("[]") == ok(list(@~[]));
        assert from_str("[ ]") == ok(list(@~[]));
        assert from_str("[true]") == ok(list(@~[boolean(true)]));
        assert from_str("[ false ]") == ok(list(@~[boolean(false)]));
        assert from_str("[null]") == ok(list(@~[null]));
        assert from_str("[3, 1]") == ok(list(@~[num(3f), num(1f)]));
        assert from_str("\n[3, 2]\n") == ok(list(@~[num(3f), num(2f)]));
        assert from_str("[2, [4, 1]]") ==
               ok(list(@~[num(2f), list(@~[num(4f), num(1f)])]));
    }

    #[test]
    fn test_read_dict() {
        assert from_str("{") ==
            err({line: 1u, col: 2u, msg: @"EOF while parsing object"});
        assert from_str("{ ") ==
            err({line: 1u, col: 3u, msg: @"EOF while parsing object"});
        assert from_str("{1") ==
            err({line: 1u, col: 2u, msg: @"key must be a string"});
        assert from_str("{ \"a\"") ==
            err({line: 1u, col: 6u, msg: @"EOF while parsing object"});
        assert from_str("{\"a\"") ==
            err({line: 1u, col: 5u, msg: @"EOF while parsing object"});
        assert from_str("{\"a\" ") ==
            err({line: 1u, col: 6u, msg: @"EOF while parsing object"});

        assert from_str("{\"a\" 1") ==
            err({line: 1u, col: 6u, msg: @"expecting ':'"});
        assert from_str("{\"a\":") ==
            err({line: 1u, col: 6u, msg: @"EOF while parsing value"});
        assert from_str("{\"a\":1") ==
            err({line: 1u, col: 7u, msg: @"EOF while parsing object"});
        assert from_str("{\"a\":1 1") ==
            err({line: 1u, col: 8u, msg: @"expecting ',' or '}'"});
        assert from_str("{\"a\":1,") ==
            err({line: 1u, col: 8u, msg: @"EOF while parsing object"});

        assert eq(result::get(from_str("{}")), mk_dict(~[]));
        assert eq(result::get(from_str("{\"a\": 3}")),
                  mk_dict(~[("a", num(3.0f))]));

        assert eq(result::get(from_str("{ \"a\": null, \"b\" : true }")),
                  mk_dict(~[
                      ("a", null),
                      ("b", boolean(true))]));
        assert eq(result::get(from_str("\n{ \"a\": null, \"b\" : true }\n")),
                  mk_dict(~[
                      ("a", null),
                      ("b", boolean(true))]));
        assert eq(result::get(from_str("{\"a\" : 1.0 ,\"b\": [ true ]}")),
                  mk_dict(~[
                      ("a", num(1.0)),
                      ("b", list(@~[boolean(true)]))
                  ]));
        assert eq(result::get(from_str(
                      "{" +
                          "\"a\": 1.0, " +
                          "\"b\": [" +
                              "true," +
                              "\"foo\\nbar\", " +
                              "{ \"c\": {\"d\": null} } " +
                          "]" +
                      "}")),
                  mk_dict(~[
                      ("a", num(1.0f)),
                      ("b", list(@~[
                          boolean(true),
                          string(@"foo\nbar"),
                          mk_dict(~[
                              ("c", mk_dict(~[("d", null)]))
                          ])
                      ]))
                  ]));
    }

    #[test]
    fn test_multiline_errors() {
        assert from_str("{\n  \"foo\":\n \"bar\"") ==
            err({line: 3u, col: 8u, msg: @"EOF while parsing object"});
    }
}
