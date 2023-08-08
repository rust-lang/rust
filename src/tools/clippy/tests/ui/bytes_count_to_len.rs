//@run-rustfix
#![warn(clippy::bytes_count_to_len)]
use std::fs::File;
use std::io::Read;

fn main() {
    // should fix, because type is String
    let _ = String::from("foo").bytes().count();

    let s1 = String::from("foo");
    let _ = s1.bytes().count();

    // should fix, because type is &str
    let _ = "foo".bytes().count();

    let s2 = "foo";
    let _ = s2.bytes().count();

    // make sure using count() normally doesn't trigger warning
    let vector = [0, 1, 2];
    let _ = vector.iter().count();

    // The type is slice, so should not fix
    let _ = &[1, 2, 3].bytes().count();

    let bytes: &[u8] = &[1, 2, 3];
    bytes.bytes().count();

    // The type is File, so should not fix
    let _ = File::open("foobar").unwrap().bytes().count();

    let f = File::open("foobar").unwrap();
    let _ = f.bytes().count();
}
