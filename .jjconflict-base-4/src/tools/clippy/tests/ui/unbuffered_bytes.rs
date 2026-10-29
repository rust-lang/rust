#![warn(clippy::unbuffered_bytes)]

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Stdin, stdin};
use std::net::TcpStream;

fn main() {
    // File is not buffered, should complain
    let file = File::open("./bytes.txt").unwrap();
    file.bytes();
    //~^ unbuffered_bytes

    // TcpStream is not buffered, should complain
    let tcp_stream: TcpStream = TcpStream::connect("127.0.0.1:80").unwrap();
    tcp_stream.bytes();
    //~^ unbuffered_bytes

    // BufReader<File> is buffered, should not complain
    let file = BufReader::new(File::open("./bytes.txt").unwrap());
    file.bytes();

    // Cursor is buffered, should not complain
    let cursor = Cursor::new(Vec::new());
    cursor.bytes();

    // Stdio would acquire the lock for every byte, should complain
    let s: Stdin = stdin();
    s.bytes();
    //~^ unbuffered_bytes

    // But when locking stdin, this is fine so should not complain
    let s: Stdin = stdin();
    let s = s.lock();
    s.bytes();
}

fn use_read<R: Read>(r: R) {
    // Callers of `use_read` may choose a `R` that is not buffered
    r.bytes();
    //~^ unbuffered_bytes
}
