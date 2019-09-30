// ignore-windows: File handling is not implemented yet
// compile-flags: -Zmiri-disable-isolation

use std::fs::File;
use std::io::{ Read, Write };

fn main() {
    // FIXME: remove the file and delete it when `rm` is implemented.

    // Test creating, writing and closing a file (closing is tested when `file` is dropped).
    let mut file = File::create("./tests/hello.txt").unwrap();
    file.write(b"Hello, World!\n").unwrap();

    // Test opening, reading and closing a file.
    let mut file = File::open("./tests/hello.txt").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    assert_eq!("Hello, World!\n", contents);
}
