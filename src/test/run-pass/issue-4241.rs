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
use std::isize;

use std::io::{ReaderUtil,WriterUtil};

enum Result {
  Nil,
  Int(isize),
  Data(~[u8]),
  List(~[Result]),
  Error(String),
  Status(String)
}

priv fn parse_data(len: usize, io: @io::Reader) -> Result {
  let res =
      if (len > 0) {
      let bytes = io.read_bytes(len as usize);
      assert_eq!(bytes.len(), len);
      Data(bytes)
  } else {
      Data(~[])
  };
  assert_eq!(io.read_char(), '\r');
  assert_eq!(io.read_char(), '\n');
  return res;
}

priv fn parse_list(len: usize, io: @io::Reader) -> Result {
    let mut list: ~[Result] = ~[];
    for _ in 0..len {
        let v = match io.read_char() {
            '$' => parse_bulk(io),
            ':' => parse_int(io),
             _ => panic!()
        };
        list.push(v);
    }
    return List(list);
}

priv fn chop(s: String) -> String {
  s.slice(0, s.len() - 1).to_string()
}

priv fn parse_bulk(io: @io::Reader) -> Result {
    match from_str::<isize>(chop(io.read_line())) {
    None => panic!(),
    Some(-1) => Nil,
    Some(len) if len >= 0 => parse_data(len as usize, io),
    Some(_) => panic!()
    }
}

priv fn parse_multi(io: @io::Reader) -> Result {
    match from_str::<isize>(chop(io.read_line())) {
    None => panic!(),
    Some(-1) => Nil,
    Some(0) => List(~[]),
    Some(len) if len >= 0 => parse_list(len as usize, io),
    Some(_) => panic!()
    }
}

priv fn parse_int(io: @io::Reader) -> Result {
    match from_str::<isize>(chop(io.read_line())) {
    None => panic!(),
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
    _ => panic!()
    }
}

priv fn cmd_to_string(cmd: ~[String]) -> String {
  let mut res = "*".to_string();
  res.push_str(cmd.len().to_string());
  res.push_str("\r\n");
    for s in &cmd {
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
