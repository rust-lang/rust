// ignore-windows: File handling is not implemented yet
// compile-flags: -Zmiri-disable-isolation

use std::fs::{File, remove_file};
use std::io::{Read, Write, ErrorKind};

fn main() {
    let path = std::env::temp_dir().join("miri_test_fs.txt");
    let bytes = b"Hello, World!\n";
    // Test creating, writing and closing a file (closing is tested when `file` is dropped).
    let mut file = File::create(&path).unwrap();
    // Writing 0 bytes should not change the file contents.
    file.write(&mut []).unwrap();

    file.write(bytes).unwrap();
    // Test opening, reading and closing a file.
    let mut file = File::open(&path).unwrap();
    let mut contents = Vec::new();
    // Reading 0 bytes should not move the file pointer.
    file.read(&mut []).unwrap();
    // Reading until EOF should get the whole text.
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes, contents.as_slice());
    // Removing file should succeed
    remove_file(&path).unwrap();

    // The two following tests also check that the `__errno_location()` shim is working properly.
    // Opening a non-existing file should fail with a "not found" error.
    assert_eq!(ErrorKind::NotFound, File::open(&path).unwrap_err().kind());
    // Removing a non-existing file should fail with a "not found" error.
    assert_eq!(ErrorKind::NotFound, remove_file(&path).unwrap_err().kind());
}
