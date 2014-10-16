// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test needs networking

extern crate extra;

use extra::net::tcp::TcpSocketBuf;

use std::io;
use std::int;

use std::io::{ReaderUtil,WriterUtil};

enum Result {
  Nil,
  Int(int),
  Data(~[u8]),
  List(~[Result]),
  Error(String),
  Status(String)
}

priv fn parse_data(len: uint, io: @io::Reader) -> Result {
  let res =
      if (len > 0) {
      let bytes = io.read_bytes(len as uint);
      assert_eq!(bytes.len(), len);
      Data(bytes)
  } else {
      Data(~[])
  };
  assert_eq!(io.read_char(), '\r');
  assert_eq!(io.read_char(), '\n');
  return res;
}

priv fn parse_list(len: uint, io: @io::Reader) -> Result {
    let mut list: ~[Result] = ~[];
    for _ in range(0, len) {
        let v = match io.read_char() {
            '$' => parse_bulk(io),
            ':' => parse_int(io),
             _ => fail!()
        };
        list.push(v);
    }
    return List(list);
}

priv fn chop(s: String) -> String {
  s.slice(0, s.len() - 1).to_string()
}

priv fn parse_bulk(io: @io::Reader) -> Result {
    match from_str::<int>(chop(io.read_line())) {
    None => fail!(),
    Some(-1) => Nil,
    Some(len) if len >= 0 => parse_data(len as uint, io),
    Some(_) => fail!()
    }
}

priv fn parse_multi(io: @io::Reader) -> Result {
    match from_str::<int>(chop(io.read_line())) {
    None => fail!(),
    Some(-1) => Nil,
    Some(0) => List(~[]),
    Some(len) if len >= 0 => parse_list(len as uint, io),
    Some(_) => fail!()
    }
}

priv fn parse_int(io: @io::Reader) -> Result {
    match from_str::<int>(chop(io.read_line())) {
    None => fail!(),
    Some(i) => Int(i)
    }
}

priv fn parse_response(io: @io::Reader) -> Result {
    match io.read_char() {
    '$' => parse_bulk(io),
    '*' => parse_multi(io),
    '+' => Status(chop(io.read_line())),
    '-' => Error(chop(io.read_line())),
    ':' => parse_int(io),
    _ => fail!()
    }
}

priv fn cmd_to_string(cmd: ~[String]) -> String {
  let mut res = "*".to_string();
  res.push_str(cmd.len().to_string());
  res.push_str("\r\n");
    for s in cmd.iter() {
    res.push_str(["$".to_string(), s.len().to_string(), "\r\n".to_string(),
                  (*s).clone(), "\r\n".to_string()].concat() );
    }
  res
}

fn query(cmd: ~[String], sb: TcpSocketBuf) -> Result {
  let cmd = cmd_to_string(cmd);
  //println!("{}", cmd);
  sb.write_str(cmd);
  let res = parse_response(@sb as @io::Reader);
  res
}

fn query2(cmd: ~[String]) -> Result {
  let _cmd = cmd_to_string(cmd);
    io::with_str_reader("$3\r\nXXX\r\n".to_string())(|sb| {
    let res = parse_response(@sb as @io::Reader);
    println!("{}", res);
    res
    });
}


pub fn main() {
}
