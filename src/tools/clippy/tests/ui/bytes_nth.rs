// run-rustfix

#![allow(clippy::unnecessary_operation)]
#![warn(clippy::bytes_nth)]

fn main() {
    let s = String::from("String");
    s.bytes().nth(3);
    let _ = &s.bytes().nth(3);
    s[..].bytes().nth(3);
}
