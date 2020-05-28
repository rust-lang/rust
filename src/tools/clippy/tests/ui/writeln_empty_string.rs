// run-rustfix

#![allow(unused_must_use)]
#![warn(clippy::writeln_empty_string)]
use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // These should fail
    writeln!(&mut v, "");

    let mut suggestion = Vec::new();
    writeln!(&mut suggestion, "");

    // These should be fine
    writeln!(&mut v);
    writeln!(&mut v, " ");
    write!(&mut v, "");
}
