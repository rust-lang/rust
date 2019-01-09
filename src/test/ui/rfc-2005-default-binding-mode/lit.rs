// FIXME(tschottdorf): we want these to compile, but they don't.

fn with_str() {
    let s: &'static str = "abc";

    match &s {
            "abc" => true, //~ ERROR mismatched types
            _ => panic!(),
    };
}

fn with_bytes() {
    let s: &'static [u8] = b"abc";

    match &s {
        b"abc" => true, //~ ERROR mismatched types
        _ => panic!(),
    };
}

pub fn main() {
    with_str();
    with_bytes();
}
