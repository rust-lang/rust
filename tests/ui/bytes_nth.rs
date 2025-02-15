#![allow(clippy::unnecessary_operation)]
#![allow(clippy::sliced_string_as_bytes)]
#![warn(clippy::bytes_nth)]

fn main() {
    let s = String::from("String");
    let _ = s.bytes().nth(3);
    //~^ bytes_nth
    let _ = &s.bytes().nth(3).unwrap();
    //~^ bytes_nth
    let _ = s[..].bytes().nth(3);
    //~^ bytes_nth
}
