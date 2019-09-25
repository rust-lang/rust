// ignore-windows: File handling is not implemented yet
// compile-flags: -Zmiri-disable-isolation

use std::fs::File;
use std::io::Read;

fn main() {
    // FIXME: create the file and delete it when `rm` is implemented.
    let mut file = File::open("./tests/hello.txt").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    assert_eq!("Hello, World!\n", contents);
}
