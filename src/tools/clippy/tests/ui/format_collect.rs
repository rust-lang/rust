#![allow(unused, dead_code)]
#![warn(clippy::format_collect)]

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect()
    //~^ ERROR: use of `format!` to build up a string from an iterator
}

#[rustfmt::skip]
fn hex_encode_deep(bytes: &[u8]) -> String {
    bytes.iter().map(|b| {{{{{ format!("{b:02X}") }}}}}).collect()
    //~^ ERROR: use of `format!` to build up a string from an iterator
}

macro_rules! fmt {
    ($x:ident) => {
        format!("{x:02X}", x = $x)
    };
}

fn from_macro(bytes: &[u8]) -> String {
    bytes.iter().map(|x| fmt!(x)).collect()
}

fn with_block() -> String {
    (1..10)
        //~^ ERROR: use of `format!` to build up a string from an iterator
        .map(|s| {
            let y = 1;
            format!("{s} {y}")
        })
        .collect()
}
fn main() {}
