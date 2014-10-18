// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::BufReader;
use std::io::BufferedReader;
use std::io::File;
use std::io::IoResult;

struct Lexer<R: Reader>
{
    reader: BufferedReader<R>,
}

impl<R: Reader> Lexer<R>
{
    pub fn new_from_reader(r: R) -> Lexer<R>
    {
        Lexer{reader: BufferedReader::new(r)}
    }

    pub fn new_from_file(p: Path) -> IoResult<Lexer<File>>
    {
        Ok(Lexer::new_from_reader(try!(File::open(&p))))
    }

    pub fn new_from_str<'a>(s: &'a str) -> Lexer<BufReader<'a>>
    {
        Lexer::new_from_reader(BufReader::new(s.as_bytes()))
    }
}

fn main() {}
