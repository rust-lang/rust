// run-rustfix

#![allow(dead_code, unused_variables)]
#![warn(clippy::string_lit_as_bytes)]

fn str_lit_as_bytes() {
    let bs = "hello there".as_bytes();

    let bs = r###"raw string with three ### in it and some " ""###.as_bytes();

    // no warning, because this cannot be written as a byte string literal:
    let ubs = "☃".as_bytes();

    let strify = stringify!(foobar).as_bytes();

    let includestr = include_str!("entry.rs").as_bytes();
}

fn main() {}
