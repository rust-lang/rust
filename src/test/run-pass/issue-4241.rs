// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// xfail-test needs networking

extern mod extra;

use extra::net::tcp::TcpSocketBuf;

use std::io;
use std::int;

use std::io::{ReaderUtil,WriterUtil};

enum Result {
  Nil,
  Int(int),
  Data(~[u8]),
  List(~[Result]),
  Error(~str),
  Status(~str)
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
    do len.times {
    let v =
        match io.read_char() {
        '$' => parse_bulk(io),
        ':' => parse_int(io),
         _ => fail!()
    };
    list.push(v);
    }
  return List(list);
}

priv fn chop(s: ~str) -> ~str {
  s.slice(0, s.len() - 1).to_owned()
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

priv fn cmd_to_str(cmd: ~[~str]) -> ~str {
  let mut res = ~"*";
  res.push_str(cmd.len().to_str());
  res.push_str("\r\n");
    for s in cmd.iter() {
    res.push_str([~"$", s.len().to_str(), ~"\r\n",
                  (*s).clone(), ~"\r\n"].concat() );
    }
  res
}

fn query(cmd: ~[~str], sb: TcpSocketBuf) -> Result {
  let cmd = cmd_to_str(cmd);
  //io::println(cmd);
  sb.write_str(cmd);
  let res = parse_response(@sb as @io::Reader);
  //printfln!(res);
  res
}

fn query2(cmd: ~[~str]) -> Result {
  let _cmd = cmd_to_str(cmd);
    do io::with_str_reader(~"$3\r\nXXX\r\n") |sb| {
    let res = parse_response(@sb as @io::Reader);
    printfln!(res);
    res
    }
}


fn main() {
}
