// run-rustfix

#![allow(dead_code, unused_variables)]
#![warn(clippy::string_lit_as_bytes)]

fn str_lit_as_bytes() {
    let bs = "hello there".as_bytes();

    let bs = r###"raw string with 3# plus " ""###.as_bytes();

    // no warning, because these cannot be written as byte string literals:
    let ubs = "☃".as_bytes();
    let ubs = "hello there! this is a very long string".as_bytes();

    let strify = stringify!(foobar).as_bytes();

    let includestr = include_str!("entry.rs").as_bytes();
}

fn main() {}
