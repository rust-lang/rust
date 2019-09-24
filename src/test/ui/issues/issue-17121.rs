// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616
// ignore-cloudabi no std::fs

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
