// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::{Buffer, IoResult, IoError, OtherIoError};
use std::char::is_alphanumeric;
use std::collections::hashmap::HashMap;

#[deriving(PartialEq, Show, Clone)]
pub enum Token {
    Noop,
    Matter,
    Colon,
    Ident(String),
    TString(String)
}

#[deriving(Hash, PartialEq, Show)]
pub enum Value {
    VString(String),
    VBool(bool),
    VInt(int),
    VUint(uint)
}

pub struct Frontmatter<R> {
    lexer: Lexer<R>,
    pub keys: HashMap<String, Value>
}

impl<R: Buffer> Frontmatter<R> {
    pub fn new(buf: R) -> Frontmatter<R> {
        Frontmatter {
            lexer: Lexer::new(buf),
            keys: HashMap::new()
        }
    }

    pub fn expect(&mut self, tok: Token) -> bool {
        match self.lexer.bump() {
            Ok(ref t) if t == &tok => true,
            _ => false
        }
    }

    pub fn parse_pair(&mut self, key: String) -> Result<(), &'static str> {
        if !self.expect(Colon) {
            return Err("Expected a colon");
        }

        match self.lexer.bump() {
            Ok(ref v) => {
                let val = match v {
                    &TString(ref s) => VString(s.clone()),
                    _ => return Err("Further types have not been implemented yet.")
                };
                self.keys.insert(key, val);
            },
            _ => return Err("Failed!")
        }

        Ok(())
    }

    pub fn parse_keys(&mut self) -> Result<(), &'static str> {

        loop {
            match self.lexer.bump() {
                Ok(Ident(f)) => {
                    try!(self.parse_pair(f));
                },
                _ => break
            }
        }

        Ok(())
    }

    pub fn parse(&mut self) -> Result<Vec<u8>, &'static str> {
        if !self.expect(Matter) {
            return Err("Expected a matter token.");
        }

        try!(self.parse_keys());

        if !self.expect(Matter) {
            return Err("Expected a matter token.");
        }

        Ok(self.lexer.buf.read_to_end().unwrap())
    }
}

pub struct Lexer<R> {
    buf: R,
    next_token: Option<Token>
}

impl<R: Buffer> Lexer<R> {
    pub fn new(buf: R) -> Lexer<R> {
        Lexer {
            buf: buf,
            next_token: None
        }
    }

    pub fn bump(&mut self) -> IoResult<Token> {
        if self.next_token.is_some() {
            return Ok(self.next_token.take().unwrap());
        }

        let ch = try!(self.buf.read_char());

        match ch {
            '-' => {
                let line = try!(self.buf.read_line());
                if line.as_slice() == "--" || line.as_slice() == "--\n" {
                    return Ok(Matter);
                }
            },
            ':' => return Ok(Colon),
            '\n' => { return self.bump() },
            ' ' => { return self.bump() },
            '"' => {
                let mut val = String::new();

                loop {
                    let next_ch = try!(self.buf.read_char());

                    if next_ch == '"' {
                        break;
                    } else {
                        val.push_char(next_ch);
                    }
                }

                return Ok(TString(val));
            },
            'A'..'Z' | 'a'..'z' => {
                let mut ident = String::new();
                ident.push_char(ch);

                loop {
                    let next_ch = try!(self.buf.read_char());
                    if is_alphanumeric(next_ch) {
                        ident.push_char(next_ch);
                        continue;
                    } else if next_ch == ':' {
                        self.next_token = Some(Colon);
                    }

                    break;
                }

                return Ok(Ident(ident));
            },
            _ => {}
        }

        Err(IoError { kind: OtherIoError, desc: "Invalid token", detail: None })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::MemReader;

    #[test]
    fn empty_source() {
        let buf = MemReader::new(vec![]);
        let mut lex = Lexer::new(buf);

        assert!(lex.bump().is_err());
    }

    #[test]
    fn matter_one() {
        let src = String::from_str("---").into_bytes();
        let buf = MemReader::new(src);
        let mut lex = Lexer::new(buf);

        assert_eq!(lex.bump().unwrap(), Matter);
    }

    #[test]
    fn matter_two() {
        let src = String::from_str("---\n---").into_bytes();
        let buf = MemReader::new(src);
        let mut lex = Lexer::new(buf);

        assert_eq!(lex.bump().unwrap(), Matter);
        assert_eq!(lex.bump().unwrap(), Matter);
    }

    #[test]
    fn ident() {
        let src = String::from_str("---\nfoon\n---").into_bytes();
        let buf = MemReader::new(src);
        let mut lex = Lexer::new(buf);

        assert_eq!(lex.bump().unwrap(), Matter);
        assert_eq!(lex.bump().unwrap(), Ident("foon".to_string()));
        assert_eq!(lex.bump().unwrap(), Matter);
    }

    #[test]
    fn ident_colon() {
        let src = String::from_str("---\nfoon:\n---").into_bytes();
        let buf = MemReader::new(src);
        let mut lex = Lexer::new(buf);

        assert_eq!(lex.bump().unwrap(), Matter);
        assert_eq!(lex.bump().unwrap(), Ident("foon".to_string()));
        assert_eq!(lex.bump().unwrap(), Colon);
        assert_eq!(lex.bump().unwrap(), Matter);
    }

    #[test]
    fn ident_colon_value_str() {
        let src = String::from_str("---\nfoon: \"bar\"\n---").into_bytes();
        let buf = MemReader::new(src);
        let mut lex = Lexer::new(buf);

        assert_eq!(lex.bump().unwrap(), Matter);
        assert_eq!(lex.bump().unwrap(), Ident("foon".to_string()));
        assert_eq!(lex.bump().unwrap(), Colon);
        assert_eq!(lex.bump().unwrap(), TString("bar".to_string()));
        assert_eq!(lex.bump().unwrap(), Matter);
    }

    #[test]
    fn ident_colon_value_multiline_str() {
        let src = String::from_str("---\nfoon: \"b\nar\"\n---").into_bytes();
        let buf = MemReader::new(src);
        let mut lex = Lexer::new(buf);

        assert_eq!(lex.bump().unwrap(), Matter);
        assert_eq!(lex.bump().unwrap(), Ident("foon".to_string()));
        assert_eq!(lex.bump().unwrap(), Colon);
        assert_eq!(lex.bump().unwrap(), TString("b\nar".to_string()));
        assert_eq!(lex.bump().unwrap(), Matter);
    }

    #[test]
    fn ident_colon_value_bool() {
        let src = String::from_str("---\nfoon: true\n---").into_bytes();
        let buf = MemReader::new(src);
        let mut lex = Lexer::new(buf);

        assert_eq!(lex.bump().unwrap(), Matter);
        assert_eq!(lex.bump().unwrap(), Ident("foon".to_string()));
        assert_eq!(lex.bump().unwrap(), Colon);
        assert_eq!(lex.bump().unwrap(), Ident("true".to_string()));
        assert_eq!(lex.bump().unwrap(), Matter);
    }

    #[test]
    fn parse_keyval() {
        let src = String::from_str("---\nfoon: \"bar\"\n---").into_bytes();
        let buf = MemReader::new(src);
        let mut matter = Frontmatter::new(buf);
        matter.parse();

        assert_eq!(matter.keys.get(&"foon".to_string()), &VString("bar".to_string()));
    }

    #[test]
    fn parse_multi_keyval() {
        let src = String::from_str("---\nfoon: \"bar\"\ntitle: \"Something\"\n---").into_bytes();
        let buf = MemReader::new(src);
        let mut matter = Frontmatter::new(buf);
        matter.parse();

        assert_eq!(matter.keys.get(&"foon".to_string()), &VString("bar".to_string()));
        assert_eq!(matter.keys.get(&"title".to_string()), &VString("Something".to_string()));
    }
}
