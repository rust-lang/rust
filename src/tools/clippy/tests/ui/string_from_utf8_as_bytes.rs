#![warn(clippy::string_from_utf8_as_bytes)]

macro_rules! test_range {
    ($start:expr, $end:expr) => {
        $start..$end
    };
}

fn main() {
    let _ = std::str::from_utf8(&"Hello World!".as_bytes()[6..11]);
    //~^ string_from_utf8_as_bytes

    let s = "Hello World!";
    let _ = std::str::from_utf8(&s.as_bytes()[test_range!(6, 11)]);
    //~^ string_from_utf8_as_bytes
}
