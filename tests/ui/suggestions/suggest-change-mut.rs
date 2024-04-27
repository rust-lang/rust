#![allow(warnings)]

use std::io::{BufRead, BufReader, Read, Write};

fn issue_81421<T: Read + Write>(mut stream: T) { //~ HELP consider introducing a `where` clause
    let initial_message = format!("Hello world");
    let mut buffer: Vec<u8> = Vec::new();
    let bytes_written = stream.write_all(initial_message.as_bytes());
    let flush = stream.flush();

    loop {
        let mut stream_reader = BufReader::new(&stream);
        //~^ ERROR the trait bound `&T: std::io::Read` is not satisfied [E0277]
        //~| HELP consider removing the leading `&`-reference
        //~| HELP consider changing this borrow's mutability
        stream_reader.read_until(b'\n', &mut buffer).expect("Reading into buffer failed");
        //~^ ERROR the method `read_until` exists for struct `BufReader<&T>`,
    }
}

fn main() {}
