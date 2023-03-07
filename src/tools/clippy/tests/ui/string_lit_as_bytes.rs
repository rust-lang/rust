// run-rustfix

#![allow(dead_code, unused_variables)]
#![warn(clippy::string_lit_as_bytes)]

fn str_lit_as_bytes() {
    let bs = "hello there".as_bytes();

    let bs = r###"raw string with 3# plus " ""###.as_bytes();

    let bs = "lit to string".to_string().into_bytes();
    let bs = "lit to owned".to_owned().into_bytes();

    // no warning, because these cannot be written as byte string literals:
    let ubs = "☃".as_bytes();
    let ubs = "hello there! this is a very long string".as_bytes();

    let ubs = "☃".to_string().into_bytes();
    let ubs = "this is also too long and shouldn't be fixed".to_string().into_bytes();

    let strify = stringify!(foobar).as_bytes();

    let current_version = env!("CARGO_PKG_VERSION").as_bytes();

    let includestr = include_str!("string_lit_as_bytes.rs").as_bytes();

    let _ = "string with newline\t\n".as_bytes();

    let _ = match "x".as_bytes() {
        b"xx" => 0,
        [b'x', ..] => 1,
        _ => 2,
    };
}

fn main() {}
