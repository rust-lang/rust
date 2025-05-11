#![warn(clippy::string_from_utf8_as_bytes)]

fn main() {
    let _ = std::str::from_utf8(&"Hello World!".as_bytes()[6..11]);
    //~^ string_from_utf8_as_bytes
}
