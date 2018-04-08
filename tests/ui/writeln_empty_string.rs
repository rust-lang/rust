#![allow(unused_must_use)]
#![warn(writeln_empty_string)]
use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // This should fail
    writeln!(&mut v, "");

    // These should be fine
    writeln!(&mut v);
    writeln!(&mut v, " ");
    write!(&mut v, "");

}
