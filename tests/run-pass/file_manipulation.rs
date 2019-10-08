// ignore-windows: File handling is not implemented yet
// compile-flags: -Zmiri-disable-isolation

use std::fs::{File, remove_file};
use std::io::{Read, Write};

fn main() {
    let path = "./tests/hello.txt";
    let bytes = b"Hello, World!\n";
    // Test creating, writing and closing a file (closing is tested when `file` is dropped).
    let mut file = File::create(path).unwrap();
    // Writing 0 bytes should not change the file contents.
    file.write(&mut []).unwrap();

    file.write(bytes).unwrap();
    // Test opening, reading and closing a file.
    let mut file = File::open(path).unwrap();
    let mut contents = Vec::new();
    // Reading 0 bytes should not move the file pointer.
    file.read(&mut []).unwrap();
    // Reading until EOF should get the whole text.
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes, contents.as_slice());
    // Removing file should succeed
    remove_file(path).unwrap();
    // Opening non-existing file should fail
    assert!(File::open(path).is_err());
    // Removing non-existing file should fail
    assert!(remove_file(path).is_err());
}
