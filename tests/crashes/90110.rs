//@ known-bug: #90110

use std::fs::File;
use std::io::{BufReader, BufRead};
use std::str::Split;
use std::path::Path;

pub trait Parser<D>
where dyn Parser<D>: Sized
{
    fn new(split_header: Split<&str>) -> Self where Self: Sized;
    fn parse_line(&self, split_line: &Split<&str>) -> D;
}


pub struct CsvReader<D> {
    parser: Box<dyn Parser<D>>,

    reader: BufReader<File>,
    buf: String,    // Buffer we will read into. Avoids re-allocation on each line.
    path: String,   // Record this so we can return more informative error messages.
    line: usize,    // Same motivation for this.
}

impl<D> CsvReader<D>
where dyn Parser<D>: Sized
{
    fn new<F>(path: &str, make_parser: F) -> CsvReader<D>
    where F: Fn(Split<char>) -> dyn Parser<D> {
        let file = match File::open(Path::new(path)) {
            Err(err) => panic!("Couldn't read {}: {}", path, err),
            Ok(file) => file,
        };

        let mut reader = BufReader::new(file);

        let mut buf = String::new();

        let parser = Box::new(match reader.read_line(&mut buf) {
            Err(err) => panic!("Failed to read the header line from {}: {}", path, err),
            Ok(_) => {
                let split_header = buf.split(',');
                make_parser(split_header)
            },
        });

        CsvReader {
            parser: parser,
            reader,
            buf,
            path: path.to_string(),
            line: 2,
        }
    }
}

pub fn main() {}
