// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

use std::fs::File;
use std::io::{self, BufReader, Read};

struct Lexer<R: Read>
{
    reader: BufReader<R>,
}

impl<R: Read> Lexer<R>
{
    pub fn new_from_reader(r: R) -> Lexer<R>
    {
        Lexer{reader: BufReader::new(r)}
    }

    pub fn new_from_file(p: &str) -> io::Result<Lexer<File>>
    {
        Ok(Lexer::new_from_reader(File::open(p)?))
    }

    pub fn new_from_str<'a>(s: &'a str) -> Lexer<&'a [u8]>
    {
        Lexer::new_from_reader(s.as_bytes())
    }
}

fn main() {}
