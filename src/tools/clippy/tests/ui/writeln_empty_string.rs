#![allow(unused_must_use)]
#![warn(clippy::writeln_empty_string)]
use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // These should fail
    writeln!(v, "");
    //~^ writeln_empty_string

    let mut suggestion = Vec::new();
    writeln!(suggestion, "");
    //~^ writeln_empty_string

    // These should be fine
    writeln!(v);
    writeln!(v, " ");
    write!(v, "");
}
